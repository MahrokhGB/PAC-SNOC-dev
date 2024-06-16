from datetime import datetime
from pyro.distributions import Normal
import sys, os, math, logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from assistive_functions import WrapLogger
from experiments.robotsX.loss_functions import LossRobots
from controllers.VI_controller import VICont
from experiments.robotsX.detect_collision import *


"""
Tunable params: epsilon, prior_std, #TODO
NOTE: initialize VFs by the prior. => no need to tune initialization_std
"""

random_seed = 5
torch.manual_seed(random_seed)

# ------ EXPERIMENT ------
col_av = True
obstacle = True
is_linear = False
fname = None     # set to None to be set automatically
DEBUG = False
# -----------------------

# ----- SET UP LOGGER -----
exp_name = 'robotsX'
exp_name += '_col_av' if col_av else ''
exp_name += '_obstacle' if obstacle else ''
exp_name += '_lin' if is_linear else '_nonlin'
now = datetime.now().strftime("%m_%d_%H_%Ms")

file_path = os.path.join(BASE_DIR, 'log')
path_exist = os.path.exists(file_path)
if not path_exist:
    os.makedirs(file_path)
if fname is None:
    filename_log = exp_name+'_VI_log' + now
else:
    filename_log = fname + '_log'
filename_log = os.path.join(
    BASE_DIR, 'experiments', 'robotsX',
    'saved_results', 'log', filename_log
)

logging.basicConfig(filename=filename_log, format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('bnn')
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
    T=t_end, Q=Q, alpha_u=alpha_u, xbar=xbar,
    loss_bound=loss_bound, sat_bound=sat_bound.to(device),
    alpha_ca=alpha_ca, alpha_obst=alpha_obst,
    min_dist=min_dist if col_av else None,
    n_agents=sys.n_agents if col_av else None,
    num_states=sys.num_states if col_av else None
)
original_loss_fn = LossRobots(
    T=t_end, Q=Q, alpha_u=alpha_u, xbar=xbar,
    loss_bound=None, sat_bound=None,
    alpha_ca=alpha_ca, alpha_obst=alpha_obst,
    min_dist=min_dist if col_av else None,
    n_agents=sys.n_agents if col_av else None,
    num_states=sys.num_states if col_av else None
)

# ------ 2.3. Gibbs temperature ------
epsilon = 0.1       # PAC holds with Pr >= 1-epsilon
gibbs_lambda_star = (8*num_rollouts*math.log(1/epsilon))**0.5   # lambda for Gibbs
gibbs_lambda = gibbs_lambda_star

# ------ 2.4. define the prior ------
prior_std = 7
prior_dict = {'type':'Gaussian'}
training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
var_prior_dict = dict.fromkeys(
    [t+'_vec' for t in training_param_names]
)
for name in training_param_names:
    prior_dict[name+'_vec_loc'] = 0
    prior_dict[name+'_vec_scale'] = prior_std
    var_prior_dict[name+'_vec'] = {
        'mean': 0, 'scale': prior_std
    }

# ------ 2.5. define VI controller ------
batch_size = 5
lr = 1e-2
epochs = 3000
log_period = 50
early_stopping = True
n_xi = 8       # size of the linear part of REN
l = 8          # size of the non-linear part of REN
num_vfs=1


# OrderedDict([('out.bias', torch.Size([1])), ('out.weight', torch.Size([1]))])
# var_prior_dict ={
#     'out.bias':{
#         'mean':torch.tensor(prior_dict['bias_loc']),
#         'variance':torch.tensor(prior_dict['bias_scale']**2)
#     },
#     'out.weight':{
#         'mean':torch.tensor(prior_dict['weight_loc'].flatten()),
#         'variance':torch.tensor(prior_dict['weight_scale']**2)
#     }
# }

vi_cont = VICont(
    sys, train_d=train_data, lr=lr, loss=bounded_loss_fn,
    prior_dict=prior_dict,
    random_seed=random_seed, optimizer='Adam', batch_size=batch_size, lambda_=gibbs_lambda,
    num_iter_fit=epochs, lr_decay=0.99, logger=logger,
    # VI properties
    num_vfs=num_vfs, vf_init_std=0.1, vf_cov_type='diag',
    vf_param_dists=None,  # intialize with the prior
    # controller properties
    n_xi=n_xi, l=l, x_init=sys.x_init, u_init=sys.u_init, controller_type='REN',
    # debug
    debug=DEBUG
)

# print(vi_cont.generic_Gibbs.parameter_shapes())

# ------------ 3. Training ------------
msg = '------------------ ROBOTS X EXP------------------'
msg += '\nVI'
msg += '\n[INFO] TASK: avoid collisions: ' + str(col_av) + ', avoid obstacles: ' + str(obstacle)
msg += ', use linearized system model: ' + str(is_linear)
msg += '\n[INFO] Dataset: n_agents: %i' % n_agents + ' -- num_rollouts: %i' % num_rollouts
msg += ' -- std_ini: %.2f' % std_ini + ' -- spring k: %.2f' % k
msg += '\n[INFO] Initial condition: x_0: ' + str(x0) + ' -- xbar: ' + str(xbar)
msg += '\n[INFO] Loss: t_end: %i'% t_end + ' -- alpha_u: %.6f' % alpha_u
msg += ' -- alpha_ca: %.f' % alpha_ca if col_av else ''
msg += ' -- alpha_obst: %.1f' % alpha_obst if obstacle else ''
msg += '\n[INFO] REN: n_xi: %i' % n_xi + ' -- l: %i' % l
msg += '\n[INFO] Solver: lr: %.4f' % lr + ' -- epochs: %i' % epochs
msg += ' -- batch_size: %i' % batch_size + ', -- early stopping:' + str(early_stopping)
msg += '\n[INFO] VI: epsilon: %.2f' % epsilon + ' -- num variational factors: %2.f' % num_vfs + ' -- gibbs_lambda: %.2f' % gibbs_lambda
msg += ' (use lambda_*)' if gibbs_lambda == gibbs_lambda_star else ''
msg += ' -- prior std: %.4f' % prior_std
logger.info(msg)

logger.info('------------ Begin training ------------')
vi_cont.fit(
    early_stopping=early_stopping, log_period=log_period
)
logger.info('Training completed.')

if DEBUG:
    for name, param in vi_cont.var_post.parameters():
        print(name, param)
    quit()

# ------ Save trained model ------
if num_vfs>1:
    raise NotImplementedError
else:
    vf_num=1
    # save this factor
    res_dict = vi_cont.var_post.parameters()
    res_dict['num_rollouts'] = num_rollouts
    res_dict['Q'], res_dict['alpha_u'] = Q, alpha_u
    res_dict['alpha_ca'], res_dict['alpha_obst'] = alpha_ca, alpha_obst
    res_dict['n_xi'], res_dict['l'] = n_xi, l
    if fname is not None:
        filename = fname+'_factor'+str(vf_num)+'.pt'
    else:
        filename = exp_name+'VIfactor'+str(vf_num)+'_T'+str(t_end)+'_S'+str(num_rollouts)
        filename += '_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pt'
    file_path = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', 'trained_models')
    path_exist = os.path.exists(file_path)
    if not path_exist:
        os.makedirs(file_path)
    filename = os.path.join(file_path, filename)
    torch.save(res_dict, filename)
    logger.info('model saved.')

# eval on train data
bounded_train_loss = vi_cont.eval_rollouts(train_data)
original_train_loss = vi_cont.eval_rollouts(train_data, loss_fn=original_loss_fn)
logger.info('Final results on the entire train data: Bounded train loss = {:.4f}, original train loss = {:.4f}'.format(
    bounded_train_loss, original_train_loss
))

# ------------ 5. Test Dataset ------------

bounded_test_loss = vi_cont.eval_rollouts(test_data)
original_test_loss = vi_cont.eval_rollouts(test_data, loss_fn=original_loss_fn)
msg = 'True bounded test loss = {:.4f}, '.format(bounded_test_loss)
msg += 'true original test loss = {:.2f} '.format(original_test_loss)
msg += '(approximated using {:3.0f} test rollouts).'.format(test_data.shape[0])
logger.info(msg)
