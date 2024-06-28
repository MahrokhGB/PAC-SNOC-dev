# Import required packages
import torch, math, tqdm
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt

import sys, os, pickle
from control import dlqr
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
logger = WrapLogger(None)
save_path = os.path.join(BASE_DIR, 'experiments', 'scalar', 'saved_results')

# ------ 1. load data ------
T = 10
S = 8
epsilon = 0.2       # PAC holds with Pr >= 1-epsilon
dist_type = 'N biased'
prior_type_b = 'Gaussian_biased_wide'

data_train, data_test, disturbance = load_data(
    dist_type=dist_type, S=S, T=T, random_seed=random_seed,
    S_test=None     # use a subset of available test data if not None
)

print(epsilon, prior_type_b, S)

# ------ 2. define the plant ------
sys = LTI_system(
    A = np.array([[0.8]]),  # num_states*num_states
    B = np.array([[0.1]]),  # num_states*num_inputs
    C = np.array([[0.3]]),  # num_outputs*num_states
    x_init = 2*np.ones((1, 1)),  # num_states*1
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
    # initialization_std=0.1, # for initializing REN. not important
)


# ****** PRIOR ******
# ------ prior on weight ------
# center prior at the infinite horizon LQR solution
K_lqr_ih, _, _ = dlqr(
    sys.A.detach().numpy(), sys.B.detach().numpy(),
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
target = GibbsWrapperNF(
target_dist=gibbs_posteior, train_data=data_train, data_batch_size=None,
    prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)
)

# load gridded Gibbs distribution
filename = dist_type.replace(" ", "_")+'_ours_'+prior_type_b+'_T'+str(T)+'_S'+str(S)+'_eps'+str(int(epsilon*10))+'.pkl'
filename = os.path.join(save_path, filename)
filehandler = open(filename, 'rb')
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
    flows += [nf.flows.Planar((2,))]

# base distribution same as the prior
q0 = nf.distributions.DiagGaussian(2)
state_dict = q0.state_dict()
state_dict['loc'] = torch.tensor(
    [prior_dict['weight_loc'], prior_dict['bias_loc']]
).reshape(1, -1)
state_dict['log_scale'] = torch.log(torch.tensor(
    [prior_dict['weight_scale'], prior_dict['bias_scale']]
)).reshape(1, -1)
q0.load_state_dict(state_dict)

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
plt.savefig(os.path.join(save_path, 'target_dist.pdf'))
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
plt.savefig(os.path.join(save_path, 'init_flow_dist.pdf'))
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
plt.savefig(os.path.join(save_path, 'init_base_dist.pdf'))
plt.show()


# ****** TRAIN NORMFLOWS ******
# Train model
max_iter = 20000
num_samples = 2 * 20
anneal_iter = 10000
annealing = True
show_iter = 2000


loss_hist = np.array([])

optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3, weight_decay=1e-4)
for it in tqdm(range(max_iter)):
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
        plt.savefig(os.path.join(save_path, name+'_flow_dist.pdf'))
        plt.show()

plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.savefig(os.path.join(save_path, 'final_loss.pdf'))
plt.show()