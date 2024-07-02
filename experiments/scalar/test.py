# Import required packages
import numpy as np
from tqdm import tqdm
import normflows as nf
from control import dlqr
from datetime import datetime
from matplotlib import pyplot as plt
import torch, math, logging, sys, os, pickle, time

BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(BASE_DIR)
from config import device
print(device)
from assistive_functions import *
from loss_functions import LQLossFH
from controllers.abstract import get_controller, CLSystem
from experiments.scalar.LTI_sys import LTI_system
from inference_algs.distributions import GibbsPosterior, GibbsWrapperNF
from experiments.scalar.scalar_assistive_functions import load_data


# ****** GENERAL ******
random_seed = 33
random_state = np.random.RandomState(random_seed)
# ----- save and log directory -----
now = datetime.now().strftime("%m_%d_%H_%Ms")
save_path = os.path.join(BASE_DIR, 'experiments', 'scalar', 'saved_results')
logger=WrapLogger(None)

# ------ 1. load data ------
T = 10
S = 4096
epsilon = 0.2       # PAC holds with Pr >= 1-epsilon
dist_type = 'N biased'
prior_type_b = 'Gaussian_biased_wide' # 'Uniform' #'Gaussian_biased_wide'

data_train, data_test, disturbance = load_data(
    dist_type=dist_type, S=S, T=T, random_seed=random_seed,
    S_test=None     # use a subset of available test data if not None
)
data_train = torch.tensor(data_train).to(device).float()


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
lq_loss_bounded = LQLossFH(Q, R, loss_bound, sat_bound)
lq_loss_original = LQLossFH(Q, R, None, None)

# ------ 4. Gibbs temperature ------
gibbs_lambda_star = (8 * S * math.log(1/epsilon))**0.5        # lambda for Gibbs

# ------ 5. controller ------
# define a generic controller
generic_controller = get_controller(
    controller_type='Affine', sys=sys,
)

cl_system = CLSystem(sys, generic_controller, random_seed)

# test rollout time
time_rollout = False
if time_rollout:
    print('\n\n\n------ CL system rollout time ------')
    repeats = 10
    times = [None]*repeats
    for size in np.logspace(3, 12, num=10, base=2):
        t = time.time()
        for num in range(repeats):
            _,_,_ = cl_system.rollout(data_train[0:int(size), :, :])
            times[num] = time.time() - t
            t = time.time()
        print('Dataset with %i samples: ' % int(size) + 'average time = %f' % (sum(times)/len(times)))



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

# test log prob likelihood time
print('\n\n\n------ Gibbs log prob likelihood time ------')
repeats = 10
times = [None]*repeats
for size in np.logspace(3, 12, num=10, base=2):
    # Wrap Gibbs distribution to be used in normflows
    data_batch_size = None #min(32, S)
    t = time.time()
    for num in range(repeats):
        z = torch.nn.Parameter(torch.zeros(1,2, device=device))
        _ = gibbs_posteior._log_prob_likelihood(z, data_train[0:int(size), :, :])
        times[num] = time.time() - t
        t = time.time()
    print('Dataset with %i samples: ' % int(size) + 'average time = %f' % (sum(times)/len(times)))


# test log prob time
time_gibbs = True
if time_gibbs:
    print('\n\n\n------ Gibbs log prob time ------')
    repeats = 10
    times = [None]*repeats
    for size in np.logspace(3, 12, num=10, base=2):
        # Wrap Gibbs distribution to be used in normflows
        data_batch_size = None #min(32, S)
        t = time.time()
        for num in range(repeats):
            z = torch.nn.Parameter(torch.zeros(1,2, device=device))
            _ = gibbs_posteior.log_prob(z, data_train[0:int(size), :, :])
            times[num] = time.time() - t
            t = time.time()
        print('Dataset with %i samples: ' % int(size) + 'average time = %f' % (sum(times)/len(times)))


# test log prob time
time_wrapped_gibbs = True
if time_wrapped_gibbs:
    print('\n\n\n------ wrapped Gibbs log prob time ------')
    repeats = 11
    times = [None]*repeats
    for size in np.logspace(3, 12, num=10, base=2):
        # Wrap Gibbs distribution to be used in normflows
        data_batch_size = None #min(32, S)
        target = GibbsWrapperNF(
            target_dist=gibbs_posteior, train_data=data_train[0:int(size), :, :],
            data_batch_size=data_batch_size,
            prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)
        )
        t = time.time()
        for num in range(repeats):
            z = torch.nn.Parameter(torch.zeros(1,2, device=device))
            _ = target.log_prob(z)
            times[num] = time.time() - t
            t = time.time()
        print('Dataset with %i samples: ' % int(size) + 'average time = %f' % (sum(times)/len(times)))
