from tqdm import tqdm
from datetime import datetime
import normflows as nf
import sys, os, math, logging
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from assistive_functions import WrapLogger
from loss_functions.robots_loss import LossRobots
from inference_algs.distributions import GibbsPosterior, GibbsWrapperNF
from experiments.robotsX.detect_collision import *


"""
Tunable params: epsilon, prior_std, num_particles
"""

random_seed = 5
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

# ------ EXPERIMENT ------
col_av = True
obstacle = True
is_linear = False
fname = None     # set to None to be set automatically
# -----------------------

# ----- SET UP LOGGER -----
exp_name = 'robotsX'
exp_name += '_col_av' if col_av else ''
exp_name += '_obstacle' if obstacle else ''
exp_name += '_lin' if is_linear else '_nonlin'

now = datetime.now().strftime("%m_%d_%H_%M")
save_path = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results')
save_folder = os.path.join(save_path, 'normflows_'+now)
os.makedirs(save_folder)
# logger
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('normflows')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)


# ------------ 1. Dataset ------------

num_rollouts = 30
t_end = 100
std_ini = 0.2
n_agents = 2
file_path = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results')
filename_data = 'data_T'+str(t_end)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'
filename_data = os.path.join(file_path, filename_data)
if not os.path.isfile(filename_data):
    print(filename_data + " does not exists.")
    print("Need to generate data!")
assert os.path.isfile(filename_data)
filehandler = open(filename_data, 'rb')
data_saved = pickle.load(filehandler)
filehandler.close()
x0 = data_saved['x0'].to(device)
xbar = data_saved['xbar'].to(device)

train_data = data_saved['train_data_full'][:num_rollouts, :, :].to(device)
assert train_data.shape[0] == num_rollouts
test_data = data_saved['test_data'].to(device)

# ------------ 2. Parameters and hyperparameters ------------

# ------ 2.1. define the plant ------
k = 1.0
# Set initial values of variables in the loop
u_init = None   # all zero
x_init = None   # same as xbar
sys = SystemRobots(xbar, x_init=x_init, u_init=u_init, k=k, is_linear=is_linear)

# ------ 2.2. define the loss ------
Q = torch.kron(torch.eye(n_agents), torch.eye(4)).to(device)
alpha_u = 0.1/400 if col_av else 1/400
alpha_ca = 100 if col_av else None
alpha_obst = 5e3 if obstacle else None
min_dist = 1.
loss_bound = 1
sat_bound = torch.matmul(torch.matmul(x0.reshape(1, -1), Q), x0.reshape(-1, 1))
sat_bound += 0 if alpha_ca is None else alpha_ca
sat_bound += 0 if alpha_obst is None else alpha_obst
sat_bound = sat_bound/20
logger.info('Loss saturates at: '+str(sat_bound))
bounded_loss_fn = LossRobots(
    Q=Q, alpha_u=alpha_u, xbar=xbar,
    loss_bound=loss_bound, sat_bound=sat_bound.to(device),
    alpha_ca=alpha_ca, alpha_obst=alpha_obst,
    min_dist=min_dist if col_av else None,
    n_agents=sys.n_agents if col_av else None,
)
original_loss_fn = LossRobots(
    Q=Q, alpha_u=alpha_u, xbar=xbar,
    loss_bound=None, sat_bound=None,
    alpha_ca=alpha_ca, alpha_obst=alpha_obst,
    min_dist=min_dist if col_av else None,
    n_agents=sys.n_agents if col_av else None,
)

# ------ 2.3. Gibbs temperature ------
epsilon = 0.1       # PAC holds with Pr >= 1-epsilon
gibbs_lambda_star = (8*num_rollouts*math.log(1/epsilon))**0.5   # lambda for Gibbs
gibbs_lambda = gibbs_lambda_star

# ------ 2.4. define the prior ------
prior_std = 7
prior_dict = {'type':'Gaussian'}
training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
for name in training_param_names:
    prior_dict[name+'_vec_loc'] = 0
    prior_dict[name+'_vec_scale'] = prior_std


# define the controller
n_xi = 8       # size of the linear part of REN
l = 8          # size of the non-linear part of REN
initialization_std = 0.1
generic_controller = get_controller(
    controller_type='REN', sys=sys,
    n_xi=n_xi, l=l, initialization_std=initialization_std,
    output_amplification=20, train_method='SVGD'
)
num_params = sum([len(p) for p in generic_controller.parameters()])
logger.info('Controller has %i parameters.' % num_params)

# ****** POSTERIOR ******
# define target distribution
gibbs_posteior = GibbsPosterior(
    loss_fn=bounded_loss_fn, lambda_=gibbs_lambda_star, prior_dict=prior_dict,
    # attributes of the CL system
    controller=generic_controller, sys=sys,
    # misc
    logger=logger,
)

# Wrap Gibbs distribution to be used in normflows
data_batch_size = None
target = GibbsWrapperNF(
    target_dist=gibbs_posteior, train_data=train_data, data_batch_size=data_batch_size,
    prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)
)


# ****** INIT NORMFLOWS ******
K = 16

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

flows = []
for i in range(K):
    # flows += [nf.flows.Radial((num_params,))]
    flows += [nf.flows.Planar((num_params,))]

# base distribution
q0 = nf.distributions.DiagGaussian(num_params)

msg = '\n[INFO] Norm flows setup: num transformations (K): %i' % K
msg += ' -- flow type: ' + str(type(flows[0])) + ' -- base dist: ' + str(type(q0))
# msg += ' -- data batch size: %i' % (data_batch_size if not data_batch_size is None else 1)
logger.info(msg)


# set up normflow
nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
nfm.to(device)


# ****** TRAIN NORMFLOWS ******
# Train model
max_iter = 100
num_samples_nf_train = 2 * 20
num_samples_nf_eval = 100
anneal_iter = 10000
annealing = True
show_iter = 10
lr = 1e-2
weight_decay = 1e-4
msg = '\n[INFO] Training setup: annealing: ' + str(annealing)
msg += ' -- annealing iter: %i' % anneal_iter if annealing else ''
msg += ' -- learning rate: %.6f' % lr + ' -- weight decay: %.6f' % weight_decay
logger.info(msg)

nf_loss_hist = [None]*max_iter

optimizer = torch.optim.Adam(nfm.parameters(), lr=lr, weight_decay=weight_decay)
with tqdm(range(max_iter)) as t:
    for it in t:
        optimizer.zero_grad()
        if annealing:
            nf_loss = nfm.reverse_kld(num_samples_nf_train, beta=min([1., 0.01 + it / anneal_iter]))
        else:
            nf_loss = nfm.reverse_kld(num_samples_nf_train)
        nf_loss.backward()
        optimizer.step()

        nf_loss_hist[it] = nf_loss.to('cpu').data.numpy()

        # Eval and log
        if (it + 1) % show_iter == 0 or it+1==max_iter:
            # sample some controllers and eval
            z, _ = nfm.sample(num_samples_nf_eval)
            print('z', z.shape)
            lpl = target.target_dist._log_prob_likelihood(params=z, train_data=train_data)

            # log nf loss
            elapsed = t.format_dict['elapsed']
            elapsed_str = t.format_interval(elapsed)
            msg = 'Iter %i' % (it+1) + ' --- elapsed time: ' + elapsed_str  + ' --- norm flow loss: %f'  % nf_loss.item()
            msg += ' --- train loss %f' % torch.mean(lpl)
            logger.info(msg)
            # save nf model
            name = 'final' if it+1==max_iter else 'itr '+str(it+1)
            torch.save(nfm.state_dict(), os.path.join(save_folder, name+'_nfm'))
            # plot loss
            plt.figure(figsize=(10, 10))
            plt.plot(nf_loss_hist, label='loss')
            plt.legend()
            plt.savefig(os.path.join(save_folder, 'loss.pdf'))
            plt.show()
