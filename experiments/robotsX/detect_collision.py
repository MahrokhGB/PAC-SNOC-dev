import sys, os, pickle, torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from experiments.robotsX.robots_sys import SystemRobots
from controllers.REN_controller import RENController
from config import device


def detect_collisions_singletraj(x, n_agents, min_dist):
    assert len(x.shape) == 2
    deltax = x[:, 0::4].repeat(n_agents, 1, 1) - x[:, 0::4].repeat(n_agents, 1, 1).transpose(0, 2)
    deltay = x[:, 1::4].repeat(n_agents, 1, 1) - x[:, 1::4].repeat(n_agents, 1, 1).transpose(0, 2)
    distance_sq = (deltax ** 2 + deltay ** 2)
    n_coll = ((0.0001 < distance_sq) * (distance_sq < min_dist ** 2)).sum().item()
    n_coll = 0 if n_coll is None else n_coll
    return n_coll/2

def detect_collisions_multitraj(x, n_agents, min_dist):
    assert len(x.shape) == 3
    num_cols = 0
    for i in range(x.shape[0]):
        num_cols += detect_collisions_singletraj(
            x[i, :, :], n_agents, min_dist
        )
    return num_cols

def percentage_collisions_multitraj(x, n_agents, min_dist):
    assert len(x.shape) == 3
    per_cols = 0
    for i in range(x.shape[0]):
        num_cols = detect_collisions_singletraj(
            x[i, :, :], n_agents, min_dist
        )
        if not num_cols == 0:
            per_cols += 1
    return per_cols / x.shape[0]


# calculate percentage of collisions for a trained model
if __name__ == "__main__":
    random_seed = 5

    col_av = True
    obstacle = True
    is_linear = False
    num_rollouts = 30

    exp_name = 'robotsX'
    exp_name += '_col_av' if col_av else ''
    exp_name += '_obstacle' if obstacle else ''
    exp_name += '_lin' if is_linear else '_nonlin'

    # ------ load data ------
    t_end = 100
    std_ini = 0.2
    n_agents = 2
    filename = 'data_T'+str(t_end)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'
    filename = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', filename)
    filehandler = open(filename, 'rb')
    data_saved = pickle.load(filehandler)
    filehandler.close()
    x0 = data_saved['x0'].to(device)
    xbar = data_saved['xbar'].to(device)

    train_data = data_saved['train_data_full'][:num_rollouts, :, :].to(device)
    assert train_data.shape[0]==num_rollouts
    test_data = data_saved['test_data'].to(device)

    # define the model
    k = 1.0         # spring constant
    u_init = None   # all zero
    x_init = None   # same as xbar
    sys = SystemRobots(
        xbar=xbar, x_init=x_init, u_init=u_init, is_linear=is_linear, k=k
    )

    # ------ load the SVGD controller ------
    p = 0 # particle 0
    n_xi, l = 32, 32  # NOTE: specify to load the correct model
    # Load saved model
    f_model = exp_name+'_emp_T'+str(t_end)+'_S'+str(num_rollouts)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pt'
    if f_model is None:
        raise ValueError('please specify the name of trained model you wish to load.')
    print('Loading ' + f_model)
    filename_model = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', 'trained_models', f_model)
    res_dict_particle = torch.load(filename_model)
    ctl_svgd = RENController(
        sys.noiseless_forward, num_states=sys.num_states,
        num_inputs=sys.num_inputs, output_amplification=20,
        n_xi=n_xi, l=l, x_init=sys.x_init, u_init=sys.u_init,
        initialization_std=res_dict_particle['initialization_std'],
        train_method='SVGD',
    )
    # Set state dict
    model_keys = ['X_vec', 'Y_vec', 'B2_vec', 'C2_vec', 'D21_vec', 'D22_vec', 'D12_vec']
    for model_key in model_keys:
        ctl_svgd.set_parameter(model_key, res_dict_particle[model_key].to(device))
    ctl_svgd.psi_u.eval()

    # Simulate trajectories for SVDG controller
    min_dist = 1.
    x_svdg, _, u_svgd = sys.multi_rollout(ctl_svgd, train_data)
    print('Ratio of collisions in the train set = {:.2f}'.format(percentage_collisions_multitraj(x_svdg, n_agents, min_dist)))

    x_svdg, _, u_svgd = sys.multi_rollout(ctl_svgd, test_data)
    print('Ratio of collisions in the test set = {:.2f}'.format(percentage_collisions_multitraj(x_svdg, n_agents, min_dist)))

