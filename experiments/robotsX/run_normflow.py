from tqdm import tqdm
from datetime import datetime
import normflows as nf
import sys, os, math, logging
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from plants import SystemRobots, RobotsDataset
from assistive_functions import WrapLogger
from loss_functions.robots_loss import RobotsLoss
from inference_algs.distributions import GibbsPosterior, GibbsWrapperNF
from experiments.robotsX.detect_collision import *


"""
Tunable params: epsilon, prior_std, num_particles
"""

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'minimal_example', 'saved_results')
save_folder = os.path.join(save_path, 'ren_controller_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('ren_controller_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ----- parse and set experiment arguments -----
args = argument_parser()
msg = print_args(args)
logger.info(msg)
torch.manual_seed(args.random_seed)

# ------------ 1. Dataset ------------
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant, n_agents=2)
# divide to train and test
train_data = dataset.train_data_full[:args.num_rollouts, :, :]
test_data = dataset.test_data
# data for plots
t_ext = args.horizon * 4
plot_data = torch.zeros(1, t_ext, train_data.shape[-1], device=device)
plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
plot_data = plot_data.to(device)
# batch the data
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

# ------------ 2. Plant ------------
plant_input_init = None     # all zero
plant_state_init = None     # same as xbar
sys = SystemRobots(
    xbar=dataset.xbar, x_init=plant_state_init,
    u_init=plant_input_init, linearize_plant=args.linearize_plant, k=args.spring_const
)

# ------------ 3. Controller ------------
ctl = RENController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init, output_init=sys.u_init,
    dim_internal=args.dim_internal, l=args.l,
    initialization_std=args.cont_init_std,
    output_amplification=20,
)
# plot closed-loop trajectories before training the controller
logger.info('Plotting closed-loop trajectories before training the controller...')
x_log, _, u_log = sys.rollout(ctl, plot_data)
filename = os.path.join(save_folder, 'CL_init.pdf')
plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    dataset.xbar, sys.n_agents, filename=filename, text="CL - before training", T=t_ext
)

# ------------ 4. Loss ------------
Q = torch.kron(torch.eye(args.n_agents), torch.eye(4)).to(device)   # TODO: move to args and print info
loss_bound = 1
sat_bound = torch.matmul(torch.matmul(x0.reshape(1, -1), Q), x0.reshape(-1, 1))
sat_bound += 0 if args.alpha_col is None else args.alpha_col
sat_bound += 0 if args.alpha_obst is None else args.alpha_obst
sat_bound = sat_bound/20
logger.info('Loss saturates at: '+str(sat_bound))
bounded_loss_fn = RobotsLoss(
    Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
    loss_bound=loss_bound, sat_bound=None,
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=sys.n_agents if args.col_av else None,
)
original_loss_fn = RobotsLoss(
    Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
    loss_bound=None, sat_bound=sat_bound.to(device),
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=sys.n_agents if args.col_av else None,
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
    prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0), #num_samples=num_samples_nf_train
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
max_iter = 1000
num_samples_nf_train = 40
num_samples_nf_eval = 100
anneal_iter = 10000
annealing = True
show_iter = 20
lr = 1e-2
weight_decay = 1e-4
msg = '\n[INFO] Training setup: annealing: ' + str(annealing)
msg += ' -- annealing iter: %i' % anneal_iter if annealing else ''
msg += ' -- learning rate: %.6f' % lr + ' -- weight decay: %.6f' % weight_decay
logger.info(msg)

nf_loss_hist = [None]*max_iter

torch.autograd.set_detect_anomaly(True)
optimizer = torch.optim.Adam(nfm.parameters(), lr=lr, weight_decay=weight_decay)
with tqdm(range(max_iter)) as t:
    for it in t:
        optimizer.zero_grad()
        if annealing:
            nf_loss = nfm.reverse_kld(num_samples_nf_train, beta=min([1., 0.01 + it / anneal_iter]))
        else:
            nf_loss = nfm.reverse_kld(num_samples_nf_train)
        nf_loss.backward()
        for name, param in nfm.named_parameters():
            if not torch.isfinite(param.grad).all():
                print('grad for ' + name + 'is infinite.')
            if param.isnan().any():
                print(param, ' is nan in iter ', it)
        optimizer.step()

        nf_loss_hist[it] = nf_loss.to('cpu').data.numpy()

        # Eval and log
        if (it + 1) % show_iter == 0 or it+1==max_iter:
            # sample some controllers and eval
            with torch.no_grad():
                z, _ = nfm.sample(num_samples_nf_eval)
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
