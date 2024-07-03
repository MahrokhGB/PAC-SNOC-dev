import sys, os, logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from assistive_functions import WrapLogger
from loss_functions.robots_loss import LossRobots
from loss_functions.old import OldLossRobots
from experiments.robotsX.detect_collision import *

import time
import numpy as np

random_seed = 5
torch.manual_seed(random_seed)

# ------ EXPERIMENT ------
col_av = True
obstacle = True
is_linear = False
fname = 'test'      # set to None to be determined automatically

# ----- SET UP LOGGER -----
exp_name = 'robotsX'
exp_name += '_col_av' if col_av else ''
exp_name += '_obstacle' if obstacle else ''
exp_name += '_lin' if is_linear else '_nonlin'
now = datetime.now().strftime("%m_%d_%H_%M")

file_path = os.path.join(BASE_DIR, 'log')
path_exist = os.path.exists(file_path)
if not path_exist:
    os.makedirs(file_path)
if fname is None:
    filename_log = os.path.join(file_path, exp_name+'_emp_log' + now)
else:
    filename_log = os.path.join(file_path, fname+'_log')

logging.basicConfig(filename=filename_log, format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('emp')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ------------ 1. Dataset ------------
# load data
t_end = 100
std_ini = 0.2
n_agents = 2
num_rollouts = 30

file_path = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results')
filename = 'data_T'+str(t_end)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'
filename = os.path.join(file_path, filename)
if not os.path.isfile(filename):
    print(filename + " does not exists.")
    print("Need to generate data!")
assert os.path.isfile(filename)
filehandler = open(filename, 'rb')
data_saved = pickle.load(filehandler)
filehandler.close()
x0 = data_saved['x0'].to(device)
xbar = data_saved['xbar'].to(device)

# define the dynamical system
k = 1.0         # spring constant
u_init = None   # all zero
x_init = None   # same as xbar
sys = SystemRobots(xbar=xbar, x_init=x_init, u_init=u_init, is_linear=is_linear, k=k)

#  define the cost
Q = torch.kron(torch.eye(n_agents), torch.eye(4)).to(device)
alpha_u = 0.1/400
alpha_ca = 100 if col_av else None
alpha_obst = 5e3 if obstacle else None
min_dist = 1.
loss_bound = 1
sat_bound = torch.matmul(torch.matmul(x0.reshape(1, -1), Q), x0.reshape(-1, 1))

bounded_loss_fn = LossRobots(
    Q=Q, alpha_u=alpha_u, xbar=xbar,
    loss_bound=loss_bound, sat_bound=sat_bound,
    alpha_ca=alpha_ca, alpha_obst=alpha_obst,
    min_dist=min_dist if col_av else None,
    n_agents=sys.n_agents if col_av else None,
)
old_bounded_loss_fn = OldLossRobots(
    Q=Q, alpha_u=alpha_u, xbar=xbar, T=t_end,
    loss_bound=loss_bound, sat_bound=sat_bound,
    alpha_ca=alpha_ca, alpha_obst=alpha_obst,
    min_dist=min_dist if col_av else None,
    n_agents=sys.n_agents if col_av else None,
    num_states = 8
)
# test loss time

print('\n\n\n------ average time for computing robots loss ------')
repeats = 3
times_new = [None]*repeats
times_old = [None]*repeats
x = torch.rand(512, t_end, n_agents*4).to(device)
u = torch.rand(512, t_end, 2).to(device)
for size in np.logspace(3, 9, num=7, base=2):
    for num in range(repeats):
        # loss new
        t = time.time()
        loss_new = bounded_loss_fn.forward(x[:int(size), :, :], u[:int(size), :, :])
        times_new[num] = time.time() - t
        # loss old
        t = time.time()
        loss_old = old_bounded_loss_fn.forward(x[:int(size), :, :], u[:int(size), :, :])
        times_old[num] = time.time() - t
        # check values
        assert abs(loss_new - loss_old)<=1e-5

    print('Dataset with %i samples: ' % int(size) + ' new loss = %f' % (sum(times_new)/len(times_new)) + ' old loss = %f' % (sum(times_old)/len(times_old)))

