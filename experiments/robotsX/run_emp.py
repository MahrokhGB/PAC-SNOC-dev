import sys, os, logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from assistive_functions import WrapLogger
from experiments.robotsX.loss_functions import LossRobots
from experiments.robotsX.detect_collision import *
from experiments.robotsX.plots import plot_trajectories


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
now = datetime.now().strftime("%m_%d_%H_%Ms")

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

train_data = data_saved['train_data_full'][:num_rollouts, :, :].to(device)
assert train_data.shape[0] == num_rollouts
test_data = data_saved['test_data'].to(device)
# data for plot
t_ext = t_end * 4
plot_data = torch.zeros(t_ext, train_data.shape[-1])
plot_data[0, :] = (x0.detach() - xbar)
plot_data = plot_data.to(device)

# ------------ 2. Parameters and hyperparameters ------------
n_xi = 8        # size of the linear part of REN
l = 8           # size of the non-linear part of REN

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
bounded_loss_fn = None
original_loss_fn = LossRobots(
    T=t_end, Q=Q, alpha_u=alpha_u, xbar=xbar,
    loss_bound=None, sat_bound=None,
    alpha_ca=alpha_ca, alpha_obst=alpha_obst,
    min_dist=min_dist if col_av else None,
    n_agents=sys.n_agents if col_av else None,
    num_states=sys.num_states if col_av else None
)

# define the controller
initialization_std = 0.1
ctl = RENController(
    sys.noiseless_forward, num_states=sys.num_states,
    num_inputs=sys.num_inputs, logger=logger, output_amplification=20,
    n_xi=n_xi, l=l, x_init=sys.x_init, u_init=sys.u_init,
    initialization_std=initialization_std, train_method='empirical',
)

# setup optimizer
batch_size = 5
epochs = 5000 if col_av else 100
learning_rate = 2e-3 if col_av else 5e-3
early_stopping = False       # return the best model on the validation data among all validated iteration
valid_data = train_data      # use the entire train data for validation
valid_period = 5000          # validate after every 'valid_period' iterations
assert not (valid_data is None and early_stopping)
optimizer = torch.optim.Adam(ctl.parameters(), lr=learning_rate)

# ------------ 3. Before training ------------

msg = '------------- ' + exp_name + ' EXPERIMENT - EMPIRICAL -------------'
x_log, _, u_log = sys.rollout(ctl, plot_data)
# Plots:
fname_plot = 'CL'
filename = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', fname_plot)
plot_trajectories(
    x_log[0, :, :], # remove extra dim due to batching
    xbar, sys.n_agents, exp_name=exp_name, filename=filename, text="CL - before training", T=t_ext
)
# collisions before training
num_col = detect_collisions_singletraj(
    x_log[0, :, :], # remove extra dim due to batching
    n_agents, min_dist)
msg += '\nBefore training: Number of collisions in train data = ' + str(num_col)

# ------------ 4. Training ------------

msg += '\n[INFO] Dataset: n_agents: %i' % n_agents + ' -- num_rollouts: %i' % num_rollouts
msg += ' -- std_ini: %.2f' % std_ini + ' -- spring k: %.2f' % k
msg += '\n[INFO] Initial condition: x_0: ' + str(x0) + ' -- xbar: ' + str(xbar)
msg += '\n[INFO] Loss: t_end: %i'% t_end + ' -- alpha_u: %.6f' % alpha_u
msg += ' -- alpha_ca: %.f' % alpha_ca if col_av else ''
msg += ' -- alpha_obst: %.1f' % alpha_obst if obstacle else ''
msg += ' -- loss_bound: %.2f'% loss_bound + ' -- sat_bound: %.2f'% sat_bound
msg += '\n[INFO] REN: n_xi: %i' % n_xi + ' -- l: %i' % l + ' -- initialization_std: %.2f'% initialization_std
msg += '\n[INFO] Solver: lr: %.2e' % learning_rate + ' -- epochs: %i' % epochs
msg += ' -- batch_size: %i' % batch_size + ', -- early stopping:' + str(early_stopping)
logger.info(msg)

logger.info('------------ Begin training ------------')

best_valid_loss = 1e6
best_params = None
for epoch in range(epochs):
    # batch data
    if batch_size==1:
        train_data_batch = train_data[epoch, :, :]
        train_data_batch = train_data_batch.reshape(1, *train_data_batch.shape)
    else:
        inds = torch.randperm(num_rollouts)[:batch_size]
        # NOTE: use ranperm instead of randint to avoid repeated samples in a batch
        train_data_batch = train_data[inds, :, :]

    optimizer.zero_grad()
    # simulate over t_end steps
    x_log, _, u_log = sys.multi_rollout(
        controller=ctl, data=train_data_batch, train=True,
    )
    x_log = x_log.reshape(batch_size, t_end, sys.num_states)
    u_log = u_log.reshape(batch_size, t_end, sys.num_inputs)

    # original loss of this rollout
    original_loss = original_loss_fn.forward(x_log, u_log)
    msg = 'Epoch: %i --- Original train loss: %.2f'% (epoch, original_loss)

    # compute bounded loss
    if not bounded_loss_fn is None:
        with torch.no_grad():
            bounded_loss = bounded_loss_fn.forward(x_log, u_log)
            msg += ' ---||--- Bounded train loss: %.2f' % bounded_loss

    # record state dict if best on valid
    if early_stopping and epoch%valid_period==0:
        # rollout the current controller on the calid data
        with torch.no_grad():
            x_log_valid, _, u_log_valid = sys.multi_rollout(
                controller=ctl, data=valid_data, train=False,
            )
            x_log_valid = x_log_valid.reshape(valid_data.shape[0], t_end, sys.num_states)
            u_log_valid = u_log_valid.reshape(valid_data.shape[0], t_end, sys.num_inputs)
            # original cost of the valid data
            original_loss_valid = original_loss_fn.forward(x_log_valid, u_log_valid)
        msg += ' ---||--- Original validation loss: %.2f' % (original_loss_valid.item())
        # compare with the best valid loss
        if original_loss_valid.item()<best_valid_loss:
            best_valid_loss = original_loss_valid.item()
            best_params = ctl.parameters_as_vector().detach().clone()
            msg += ' (best so far)'

    logger.info(msg)

    # Take a step
    # don't take a step at teh last epoch, b.c. o.w. last printed results and final results are different
    if epoch < epochs-1:
        original_loss.backward()
        optimizer.step()
        ctl.psi_u.set_model_param()

# ------ set to best seen during training ------
if early_stopping:
    ctl.set_parameters_as_vector(best_params)

# ------ Save trained model ------

if fname is None:
    fname = exp_name+'_emp_T'+str(t_end)+'_S'+str(num_rollouts)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pt'
elif not '.pt' in fname:
    fname = fname + '.pt'
file_path = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', 'trained_models')
path_exist = os.path.exists(file_path)
if not path_exist:
    os.makedirs(file_path)
filename = os.path.join(file_path, fname)
res_dict = ctl.psi_u.state_dict()
res_dict['num_rollouts'] = num_rollouts
res_dict['Q'], res_dict['alpha_u'] = Q, alpha_u
res_dict['alpha_ca'], res_dict['alpha_obst'] = alpha_ca, alpha_obst
res_dict['n_xi'], res_dict['l'] = n_xi, l
res_dict['initialization_std'] = initialization_std
torch.save(res_dict, filename)

logger.info('[INFO] saved trained model as: ' + fname)

# ------ results on the entire train data ------

logger.info('[INFO] evaluating the trained model on the entire train data.')
with torch.no_grad():
    x_log, _, u_log = sys.multi_rollout(
        controller=ctl, data=train_data, train=False,
    )   # use the entire train data, not a batch
    # evaluate losses
    original_loss = original_loss_fn.forward(x_log, u_log)
    msg = 'Final result: Original train loss: %.4f' % (original_loss)
    if not bounded_loss_fn is None:
        bounded_loss = bounded_loss_fn.forward(x_log, u_log)
        msg += ', Bounded train loss: %.2f' % (bounded_loss)
    logger.info(msg)

    # count collisions
    num_col = detect_collisions_multitraj(x_log, n_agents, min_dist)
    per_col = percentage_collisions_multitraj(x_log, n_agents, min_dist)
    logger.info('Number of collisions in train data = ' + str(num_col) + '. Percentage: ' + str(per_col*100) + '%')

# ------------ 5. Test Dataset ------------

logger.info('[INFO] evaluating the trained model on the test data.')
with torch.no_grad():
    # simulate over t_end steps
    x_log, _, u_log = sys.multi_rollout(
        controller=ctl, data=test_data, train=False,
    )
    # loss
    original_test_loss = original_loss_fn.forward(x_log, u_log).item()
    msg = "True original test loss : %.4f" % (original_test_loss)
    if not bounded_loss_fn is None:
        bounded_test_loss = bounded_loss_fn.forward(x_log, u_log).item()
        msg += ', Bounded test loss: %.2f' % (bounded_test_loss)
    msg += ' (approximated using {:3.0f} test rollouts).'.format(test_data.shape[0])
    logger.info(msg)

# count collisions
num_col = detect_collisions_multitraj(x_log, n_agents, min_dist)
per_col = percentage_collisions_multitraj(x_log, n_agents, min_dist)
msg += '\nNumber of collisions in test data = ' + str(num_col) + '. Percentage: ' + str(per_col*100) + '%'
logger.info(msg)
