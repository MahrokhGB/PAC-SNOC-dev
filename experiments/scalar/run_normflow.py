# Import required packages
import numpy as np
from tqdm import tqdm
import normflows as nf
from control import dlqr
from datetime import datetime
from matplotlib import pyplot as plt
import torch, math, logging, sys, os, pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(BASE_DIR)
from assistive_functions import *
from experiments.scalar.loss_functions import LQLossFH
from controllers.abstract import get_controller
from experiments.scalar.LTI_sys import LTI_system
from inference_algs.distributions import GibbsPosterior, GibbsWrapperNF
from experiments.scalar.scalar_assistive_functions import load_data


# ****** GENERAL ******
random_seed = 33
random_state = np.random.RandomState(random_seed)
# ----- save and log directory -----
now = datetime.now().strftime("%m_%d_%H_%Ms")
save_path = os.path.join(BASE_DIR, 'experiments', 'scalar', 'saved_results')
save_folder = os.path.join(save_path, 'normflows_'+now)
os.makedirs(save_folder)
# logger
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('normflows')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ------ 1. load data ------
T = 10
S = 512
epsilon = 0.2       # PAC holds with Pr >= 1-epsilon
dist_type = 'N biased'
prior_type_b = 'Gaussian_biased_wide' # 'Uniform' #'Gaussian_biased_wide'

data_train, data_test, disturbance = load_data(
    dist_type=dist_type, S=S, T=T, random_seed=random_seed,
    S_test=None     # use a subset of available test data if not None
)
msg = '------------------ SCALAR EXP ------------------'
msg += '\n[INFO] Dataset: T: %i' % T + ' -- num_rollouts: %i' % S
msg += ' -- epsilon: %.1f' % epsilon + ' -- prior over bias: ' + prior_type_b
logger.info(msg)

# ------ 2. define the plant ------
sys = LTI_system(
    A = torch.tensor([[0.8]]).to(device),  # num_states*num_states
    B = torch.tensor([[0.1]]).to(device),  # num_states*num_inputs
    C = torch.tensor([[0.3]]).to(device),  # num_outputs*num_states
    x_init = 2*torch.ones(1, 1).to(device),# num_states*1
)

# ------ 3. define the loss ------
Q = 5*torch.eye(sys.num_states).to(device)
R = 0.003*torch.eye(sys.num_inputs).to(device)
# optimal loss bound
loss_bound = 1
# sat_bound = np.matmul(np.matmul(np.transpose(sys.x_init), Q) , sys.x_init)
sat_bound = torch.matmul(torch.matmul(torch.transpose(sys.x_init, 0, 1), Q), sys.x_init)
if loss_bound is not None:
    logger.info('[INFO] bounding the loss to ' + str(loss_bound))
lq_loss_bounded = LQLossFH(Q, R, T, loss_bound, sat_bound, logger=logger)
lq_loss_original = LQLossFH(Q, R, T, None, None, logger=logger)

# ------ 4. Gibbs temperature ------
gibbs_lambda_star = (8 * S * math.log(1/epsilon))**0.5        # lambda for Gibbs

# ------ 5. controller ------
# define a generic controller
generic_controller = get_controller(
    controller_type='Affine', sys=sys,
)

# ****** PRIOR ******
# ------ prior on weight ------
# center prior at the infinite horizon LQR solution
K_lqr_ih, _, _ = dlqr(
    sys.A.detach().cpu().numpy(), sys.B.detach().cpu().numpy(),
    Q.detach().cpu().numpy(), R.detach().cpu().numpy()
)
theta_mid_grid = -K_lqr_ih[0,0]
# define prior
prior_dict = {
    'type_w':'Gaussian',
    'weight_loc':theta_mid_grid, 'weight_scale':1,
}

# ------ prior on bias ------
if prior_type_b == 'Uniform':
    prior_dict.update({
        'type_b':'Uniform',
        'bias_low':-5, 'bias_high':5
    })
elif prior_type_b == 'Gaussian_biased_wide':
    prior_dict.update({
        'type_b':'Gaussian_biased',
        'bias_loc':-disturbance['mean'][0]/sys.B[0,0],
        'bias_scale':1.5
    })


# ****** POSTERIOR ******
# define target distribution
gibbs_posteior = GibbsPosterior(
    loss_fn=lq_loss_bounded, lambda_=gibbs_lambda_star, prior_dict=prior_dict,
    # attributes of the CL system
    controller=generic_controller, sys=sys,
    # misc
    logger=logger,
)

# Wrap Gibbs distribution to be used in normflows
data_batch_size = None #min(32, S)
target = GibbsWrapperNF(
    target_dist=gibbs_posteior, train_data=data_train, data_batch_size=data_batch_size,
    prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)
)

# load gridded Gibbs distribution
filename = dist_type.replace(" ", "_")+'_ours_'+prior_type_b+'_T'+str(T)+'_S'+str(S)+'_eps'+str(int(epsilon*10))+'.pkl'
filehandler = open(os.path.join(save_path, filename), 'rb')
res_dict = pickle.load(filehandler)
filehandler.close()

res_dict['theta_grid'] = [k[0,0] for k in res_dict['theta_grid']]
res_dict['theta'] = [k[0,0] for k in res_dict['theta']]

theta_grid = np.array(res_dict['theta_grid'])
bias_grid = np.array(res_dict['bias_grid'])
Z_posterior = np.reshape(
    np.array(res_dict['posterior']),
    (len(theta_grid), len(bias_grid))
)
assert abs(sum(sum(Z_posterior))-1)<=1e-5


# ****** INIT NORMFLOWS ******
K = 16

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

flows = []
for i in range(K):
    flows += [nf.flows.Radial((2,))]
    # flows += [nf.flows.Planar((2,))]

# base distribution same as the prior
q0 = nf.distributions.DiagGaussian(2)
if prior_type_b.startswith('Gauss'):
    state_dict = q0.state_dict()
    state_dict['loc'] = torch.tensor(
        [prior_dict['weight_loc'], prior_dict['bias_loc']]
    ).reshape(1, -1)
    state_dict['log_scale'] = torch.log(torch.tensor(
        [prior_dict['weight_scale'], prior_dict['bias_scale']]
    )).reshape(1, -1)
    q0.load_state_dict(state_dict)

msg = '\n[INFO] Norm flows setup: num transformations (K): %i' % K
msg += ' -- flow type: ' + str(type(flows[0])) + ' -- base dist: ' + str(type(q0))
# msg += ' -- data batch size: %i' % (data_batch_size if not data_batch_size is None else 1)
logger.info(msg)
logger.info("{0}".format(q0.state_dict()))

# set up normflow
nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
nfm.to(device)
# only used to show vase distribution
nfm_base = nf.NormalizingFlow(q0=q0, flows=[], p=target)
nfm_base.to(device)

# plot before training
plt.figure(figsize=(10, 10))
plt.pcolormesh(bias_grid, theta_grid, Z_posterior, shading='nearest')
plt.xlabel('bias')
plt.ylabel('weight')
plt.title('target distribution')
plt.savefig(os.path.join(save_folder, 'target_dist.pdf'))
plt.show()

# Plot initial flow distribution
z, _ = nfm.sample(num_samples=2 ** 20)
z_np = z.to('cpu').data.numpy()
plt.figure(figsize=(10, 10))
plt.hist2d(
    z_np[:, 1].flatten(), z_np[:, 0].flatten(),
    (len(bias_grid), len(theta_grid)),
    range=[[bias_grid[-1], bias_grid[0]], [theta_grid[-1], theta_grid[0]]]
)
plt.xlabel('bias')
plt.ylabel('weight')
plt.title('initial flow distribution')
plt.savefig(os.path.join(save_folder, 'init_flow_dist.pdf'))
plt.show()

# Plot initial base distribution
z, _ = nfm_base.sample(num_samples=2 ** 20)
z_np = z.to('cpu').data.numpy()
plt.figure(figsize=(10, 10))
plt.hist2d(
    z_np[:, 1].flatten(), z_np[:, 0].flatten(),
    (len(bias_grid), len(theta_grid)),
    range=[[bias_grid[-1], bias_grid[0]], [theta_grid[-1], theta_grid[0]]]
)
plt.xlabel('bias')
plt.ylabel('weight')
plt.title('initial base distribution')
plt.savefig(os.path.join(save_folder, 'init_base_dist.pdf'))
plt.show()


# ****** TRAIN NORMFLOWS ******
# Train model
max_iter = 12000
num_samples = 2 * 20
anneal_iter = 10000
annealing = True
show_iter = 1000
lr = 1e-2
weight_decay = 1e-4
msg = '\n[INFO] Training setup: annealing: ' + str(annealing)
msg += ' -- annealing iter: %i' % anneal_iter if annealing else ''
msg += ' -- learning rate: %.6f' % lr + ' -- weight decay: %.6f' % weight_decay
logger.info(msg)

loss_hist = np.array([])

optimizer = torch.optim.Adam(nfm.parameters(), lr=lr, weight_decay=weight_decay)
with tqdm(range(max_iter)) as t:
    for it in t:
        optimizer.zero_grad()
        if annealing:
            loss = nfm.reverse_kld(num_samples, beta=np.min([1., 0.01 + it / anneal_iter]))
        else:
            loss = nfm.reverse_kld(num_samples)
        loss.backward()
        optimizer.step()

        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

        # Plot learned distribution
        if (it + 1) % show_iter == 0 or it+1==max_iter:
            torch.cuda.manual_seed(0)
            z, _ = nfm.sample(num_samples=2 ** 20)
            z_np = z.to('cpu').data.numpy()

            # log
            elapsed = t.format_dict['elapsed']
            elapsed_str = t.format_interval(elapsed)
            logger.info('Iter %i' % (it+1) + ' --- elapsed time: ' + elapsed_str  + ' --- loss: %f'  % loss.item())

            # save
            name = 'final' if it+1==max_iter else 'itr '+str(it+1)
            torch.save(nfm.state_dict(), os.path.join(save_folder, name+'_nfm'))

            # plot
            plt.figure(figsize=(10, 10))
            plt.hist2d(
                z_np[:, 1].flatten(), z_np[:, 0].flatten(),
                (len(bias_grid), len(theta_grid)),
                range=[[bias_grid[-1], bias_grid[0]], [theta_grid[-1], theta_grid[0]]]
            )
            plt.xlabel('bias')
            plt.ylabel('weight')
            name = 'final' if it+1==max_iter else 'itr '+str(it+1)
            plt.title(name+' flow distribution')
            plt.savefig(os.path.join(save_folder, name+'_flow_dist.pdf'))
            plt.show()

            # plot loss
            plt.figure(figsize=(10, 10))
            plt.plot(loss_hist, label='loss')
            plt.legend()

            plt.savefig(os.path.join(save_folder, 'loss.pdf'))
            plt.show()



# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

'''
rejection sampling
prop_shift and scale
why 40 params?
two batch dims
why takes longer with more data?

error with uniform prior: params go outside the support

dataset:
use torch dataset and dataloader. make sure is float

'''