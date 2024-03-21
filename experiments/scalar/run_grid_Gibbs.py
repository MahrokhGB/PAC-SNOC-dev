import math, itertools, sys, os, pickle, copy
from control import dlqr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from assistive_functions import *
from experiments.scalar.loss_functions import LQLossFH
from controllers.abstract import affine_controller
from SVGD_src.approx_upperbound import approx_upper_bound
from experiments.scalar.LTI_sys import LTI_system


random_seed = 33
random_state = np.random.RandomState(random_seed)
logger = WrapLogger(None)

# IMPORTANT CHOICES
epsilon = 0.2         # PAC holds with Pr >= 1-epsilon
prior_type_b = 'Gaussian_biased_wide'
S = 8
n_grid = 65
num_sampled_controllers = 20
print(epsilon, prior_type_b, S)

# ****** PART 1: GENERAL ******
# ------ 1. load data ------
T = 10
dist_type = 'N biased'

file_path = os.path.join(BASE_DIR, 'experiments', 'scalar', 'saved_results')
filename = dist_type.replace(" ", "_")+'_data_T'+str(T)+'_RS'+str(random_seed)+'.pkl'
filename = os.path.join(file_path, filename)
if not os.path.isfile(filename):
    print(filename + " does not exists.")
    print("Need to generate data!")
assert os.path.isfile(filename)
filehandler = open(filename, 'rb')
data_all = pickle.load(filehandler)
filehandler.close()
# divide
S_test = None   # use a subset of available test data if not None
data_train = data_all['train_big'][dist_type][:S, :, :]
if not S_test is None:
    data_test = data_all['test_big'][dist_type][:S_test, :, :]
else:
    data_test = data_all['test_big'][dist_type]
# disturbance
disturbance = data_all['disturbance']

# ------ 2. define the plant ------
sys_np = LTI_system(
    A=np.array([[0.8]]),  # num_states*num_states
    B=np.array([[0.1]]),  # num_states*num_inputs
    C=np.array([[0.3]]),  # num_outputs*num_states
    x_init=2*np.ones((1, 1)),  # num_states*1
    use_tensor=False
)
sys = LTI_system(
    sys_np.A, sys_np.B, sys_np.C, sys_np.x_init,
    use_tensor=True
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
    K_lqr_ih, _, _ = dlqr(sys_np.A, sys_np.B, Q.detach().cpu().numpy(), R.detach().cpu().numpy())
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
        'bias_loc':-disturbance['mean'][0]/sys_np.B[0,0],
        'bias_scale':1.0
    })
elif prior_type_b == 'Gaussian_biased_wide':
    prior_dict.update({
        'type_b':'Gaussian_biased',
        'bias_loc':-disturbance['mean'][0]/sys_np.B[0,0],
        'bias_scale':1.5
    })
elif prior_type_b == 'Gaussian_biased_narrow':
    prior_dict.update({
        'type_b':'Gaussian_biased',
        'bias_loc':-disturbance['mean'][0]/sys_np.B[0,0],
        'bias_scale':0.5
    })

# ****** PART 3: POSTERIOR ******

logger.info('[INFO] calculating the posterior.')
# ------ grid ------
if prior_type_w == 'Uniform':
    theta_grid = np.linspace(
        prior_dict['weight_low'], prior_dict['weight_high'], n_grid
    )
elif prior_type_w == 'Gaussian':
    theta_grid = np.linspace(
        prior_dict['weight_loc']-2,
        prior_dict['weight_loc']+2,
        n_grid
    )
else:
    raise NotImplementedError

if prior_type_b == 'Uniform':
    bias_grid = np.linspace(
        prior_dict['bias_low'], prior_dict['bias_high'],
        n_grid
    )
elif prior_type_b == 'Uniform_pos':
    bias_grid = np.linspace(-5, 5, n_grid)
    # should consider the full range, b.c. prior is on the wrong side
elif prior_type_b == 'Uniform_neg':
    n_grid = int((n_grid+1)/2)      # NOTE: range is half => half points. o.w., prior becomes half the full range
    bias_grid = np.linspace(
        prior_dict['bias_low'], prior_dict['bias_high'],
        n_grid
    )
elif prior_type_b == 'Gaussian':
    bias_grid = np.linspace(-5, 5, n_grid)
elif prior_type_b == 'Gaussian_biased_wide':
    bias_grid = np.linspace(-5, 5, n_grid)
elif prior_type_b == 'Gaussian_biased':
    n_grid = int((n_grid+1)/2)      # NOTE: range is half => half points. o.w., prior becomes half the full range
    bias_grid = np.linspace(-5, 0, n_grid)
else:
    raise NotImplementedError
theta_grid = np.flip(np.sort(theta_grid))
bias_grid = np.flip(np.sort(bias_grid))

#  ------ prior ------
if prior_type_w == 'Uniform':
    prior_w = 1/len(theta_grid)*np.ones(len(theta_grid))
elif prior_type_w == 'Gaussian':
    mean = prior_dict['weight_loc']
    sigma = prior_dict['weight_scale']
    prior_w = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(theta_grid-mean)**2/(2*sigma**2))
    prior_w = prior_w * abs(theta_grid[-1]-theta_grid[0])/len(theta_grid)
    prior_w = prior_w.flatten()
    # NOTE: convert continuous pdf to discrete histogram
    assert sum(prior_w) <= 1+1e-5, sum(prior_w)
else:
    raise NotImplementedError
if prior_type_b in ['Uniform', 'Uniform_neg']:
    prior_b = 1/len(bias_grid)*np.ones(len(bias_grid))
elif prior_type_b == 'Uniform_pos':
    prior_b = np.concatenate((
        np.zeros(int((len(bias_grid)-1)/2)),
        np.ones(int((len(bias_grid)+1)/2))
    ))/int((len(bias_grid)+1)/2)
elif prior_type_b.startswith('Gaussian'):
    mean = prior_dict['bias_loc']
    sigma = prior_dict['bias_scale']
    prior_b = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(bias_grid-mean)**2/(2*sigma**2))
    prior_b = prior_b * abs(bias_grid[-1]-bias_grid[0])/len(bias_grid)
    prior_b = prior_b.flatten()
    # NOTE: convert continuous pdf to discrete histogram
    assert sum(prior_b) <= 1+1e-5, sum(prior_b)
else:
    raise NotImplementedError

# ------ init ------
num_rows = len(theta_grid)*len(bias_grid)
res_dict = {
    # general info
    'num_rollouts':S, 'T':T, 'dist_type':dist_type,
    'prior_type_b':prior_type_b,
    # distributions
    'theta':[None]*num_rows, 'bias':[None]*num_rows,
    'prior':[a[0]*a[1] for a in itertools.product(prior_w, prior_b)],
    'posterior':[None]*num_rows,
    # evaluation
    'ub':None,
    'av_test_loss_bounded':None, 'av_test_loss_original':None,
}

# NOTE: don't divide prior by sum. grid may contain part of the prior
assert len(res_dict['prior'])==num_rows
# assert sum(res_dict['prior'])>=0.85, 'considerbale mass of the prior falls outside the grid. reduce std or enlarge the grid.'

# ------ posterior ------
for ind, (theta_tmp, bias_tmp) in enumerate(
    itertools.product(theta_grid, bias_grid)
):
    res_dict['theta'][ind] = theta_tmp
    res_dict['bias'][ind] = bias_tmp
    # define controller
    c_tmp = affine_controller(
        np.array([[theta_tmp]]), np.array([[bias_tmp]])
    )
    # roll
    x_tmp, y_tmp, u_tmp = sys_np.multi_rollout(
        c_tmp, data_train
    )
    # apply controller on train data
    train_loss_bounded = lq_loss_bounded.forward(x_tmp, u_tmp).item()
    # compute posterior unnormalized
    res_dict['posterior'][ind] = res_dict['prior'][ind] * math.exp(
        -gibbs_lambda * train_loss_bounded
    )
# NOTE: normalize
sum_posterior = sum(res_dict['posterior'])
res_dict['posterior'] = [
    x/sum_posterior for x in res_dict['posterior']
]
approximated_Z = sum_posterior

# sample
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
    controller = affine_controller(
        np.array([[sc[0]]]), np.array([[sc[1]]])
    )
    # rollout
    xs_test, ys_test, us_test = sys.multi_rollout(
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
    loss_bound=loss_bound, approximated_Z=approximated_Z
)
logger.info('Theoretical upper bound = {:1.4f}'.format(ub.item()))
if not loss_bound == 1:
    logger.info('upper bound / loss bound' + str(ub/loss_bound))
assert ub/loss_bound <= 1+(math.log(1/epsilon)/2/S)**0.5
assert ub/loss_bound > (math.log(1/epsilon)/2/S)**0.5
res_dict['ub'] = ub


# save
res_dict['theta_grid'] = theta_grid
res_dict['bias_grid'] = bias_grid
res_dict['approximated_Z'] = approximated_Z
res_dict['sampled_controllers'] = sampled_controllers
filename = dist_type.replace(" ", "_")+'_ours_'+prior_type_b+'_T'+str(T)+'_S'+str(S)+'_eps'+str(int(epsilon*10))+'.pkl'
filename = os.path.join(file_path, filename)
filehandler = open(filename, 'wb')
pickle.dump(res_dict, filehandler)
logger.info('[INFO] File saved at' + filename)
filehandler.close()
