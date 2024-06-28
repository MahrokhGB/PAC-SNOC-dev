import math, itertools, sys, os, pickle, copy
from control import dlqr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from assistive_functions import *
from experiments.scalar.loss_functions import LQLossFH
from controllers.abstract import AffineController
from inference_algs.approx_upperbound import approx_upper_bound
from experiments.scalar.LTI_sys import LTI_system
from experiments.scalar.scalar_assistive_functions import load_data, compute_posterior_by_gridding


random_seed = 33
random_state = np.random.RandomState(random_seed)
logger = WrapLogger(None)

# IMPORTANT CHOICES
epsilon = 0.2         # PAC holds with Pr >= 1-epsilon
prior_type_b = 'Gaussian_biased_wide'
n_grid = 65
num_sampled_controllers = 20


# ****** PART 1: GENERAL ******
# ------ 1. load data ------
T = 10
S = 8
dist_type = 'N biased'
data_train, data_test, disturbance = load_data(
    dist_type=dist_type, S=S, T=T, random_seed=random_seed,
    S_test=None   # use a subset of available test data if not None
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
gibbs_lambda = gibbs_lambda_star


# ****** PART 2: PRIOR ******

# ------ prior on weight ------
prior_type_w = 'Gaussian'
prior_center = 'LQR-IH'
gamma = 0
delta = 1.0  # 2.5

# get prior center
if prior_center == 'LQR-IH':
    K_lqr_ih, _, _ = dlqr(
        sys.A.detach().numpy(), sys.B.detach().numpy(),
        Q.detach().cpu().numpy(), R.detach().cpu().numpy()
    )
    theta_mid_grid = -K_lqr_ih
# define different types of prior
if prior_type_w == 'Gaussian':
    prior_dict = {
        'type_w':'Gaussian',
        'weight_loc':theta_mid_grid*(1+gamma), 'weight_scale':delta,
    }
elif prior_type_w == 'Uniform':
    prior_dict = {
        'type_w':'Uniform',
        'weight_low':theta_mid_grid*(1+gamma)-delta,
        'weight_high':theta_mid_grid*(1+gamma)+delta,
    }
elif prior_type_w == 'Gaussian_trunc':
    prior_dict = {
        'type_w':'Gaussian_trunc',
    }

# ------ prior on bias ------
if prior_type_b == 'Gaussian':
    prior_dict.update({
        'type_b':'Gaussian',
        'bias_loc':0, 'bias_scale':2.5  # 5.0,
    })
elif prior_type_b == 'Uniform':
    prior_dict.update({
        'type_b':'Uniform',
        'bias_low':-5, 'bias_high':5
    })
elif prior_type_b == 'Uniform_neg':
    prior_dict.update({
        'type_b':'Uniform_neg',
        'bias_low':-5, 'bias_high':0
    })
elif prior_type_b == 'Uniform_pos':  # wrong prior
    prior_dict.update({
        'type_b':'Uniform_pos',
        'bias_low':0, 'bias_high':5
    })
elif prior_type_b == 'Gaussian_biased':
    prior_dict.update({
        'type_b':'Gaussian_biased',
        'bias_loc':-disturbance['mean'][0]/sys.B[0,0],
        'bias_scale':1.0
    })
elif prior_type_b == 'Gaussian_biased_wide':
    prior_dict.update({
        'type_b':'Gaussian_biased',
        'bias_loc':-disturbance['mean'][0]/sys.B[0,0],
        'bias_scale':1.5
    })
elif prior_type_b == 'Gaussian_biased_narrow':
    prior_dict.update({
        'type_b':'Gaussian_biased',
        'bias_loc':-disturbance['mean'][0]/sys.B[0,0],
        'bias_scale':0.5
    })

# ****** PART 3: POSTERIOR ******
logger.info('[INFO] calculating the posterior.')
res_dict = compute_posterior_by_gridding(
    prior_dict=prior_dict, lq_loss_bounded=lq_loss_bounded,
    data_train=data_train, dist_type=dist_type,
    sys=sys, gibbs_lambda=gibbs_lambda, n_grid=n_grid
)

# sample from the posterior
sampled_inds = random_state.choice(
    range(len(res_dict['posterior'])),
    size=num_sampled_controllers,
    replace=True, p=res_dict['posterior']  # NOTE: sample with replacement
)
sampled_controllers = [
    (res_dict['theta'][i], res_dict['bias'][i]) for i in sampled_inds
]

# ****** PART 4: TEST ******
logger.info('[INFO] testing sampled controllers.')
av_test_loss_bounded = [None]*num_sampled_controllers
av_test_loss_original = [None]*num_sampled_controllers
for c_num in range(num_sampled_controllers):
    # define controller
    sc = sampled_controllers[int(c_num)]
    controller = AffineController(
        np.array([[sc[0]]]), np.array([[sc[1]]])
    )
    # rollout
    xs_test, ys_test, us_test = sys.rollout(
        controller, data_test
    )
    # test loss
    with torch.no_grad():
        av_test_loss_bounded[c_num] = lq_loss_bounded.forward(xs_test,us_test).item()
        av_test_loss_original[c_num] = lq_loss_original.forward(xs_test,us_test).item()
res_dict['av_test_loss_bounded'] = av_test_loss_bounded
res_dict['av_test_loss_original'] = av_test_loss_original


# ****** PART 5: UPPER BOUND ******
logger.info('[INFO] calculating the upper bound.')
grid_dict = copy.deepcopy(res_dict)
grid_dict = {k:v for k,v in grid_dict.items() if k in ['theta', 'bias', 'prior']}
# NOTE: use the bounded loss for approximating the upper bound
ub = approx_upper_bound(
    grid_dict=grid_dict, sys=sys,
    lq_loss=lq_loss_bounded, data=to_tensor(data_train),
    lambda_=to_tensor(gibbs_lambda), epsilon=epsilon,
    loss_bound=loss_bound, approximated_Z=res_dict['approximated_Z']
)
logger.info('Theoretical upper bound = {:1.4f}'.format(ub.item()))
if not loss_bound == 1:
    logger.info('upper bound / loss bound' + str(ub/loss_bound))
assert ub/loss_bound <= 1+(math.log(1/epsilon)/2/S)**0.5
assert ub/loss_bound > (math.log(1/epsilon)/2/S)**0.5
res_dict['ub'] = ub


# save
file_path = os.path.join(BASE_DIR, 'experiments', 'scalar', 'saved_results')
res_dict['sampled_controllers'] = sampled_controllers
filename = dist_type.replace(" ", "_")+'_ours_'+prior_type_b+'_T'+str(T)+'_S'+str(S)+'_eps'+str(int(epsilon*10))+'.pkl'
filename = os.path.join(file_path, filename)
filehandler = open(filename, 'wb')
pickle.dump(res_dict, filehandler)
logger.info('[INFO] File saved at' + filename)
filehandler.close()
