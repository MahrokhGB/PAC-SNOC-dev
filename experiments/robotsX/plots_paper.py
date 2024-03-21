import torch
import pickle
import os
import sys
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

from config import device
from experiments.robotsX.robots_sys import SystemRobots
from controllers.REN_controller import RENController
from experiments.robotsX.plots import plot_trajectories

random_seed = 5
torch.manual_seed(random_seed)

# ------ IMPORTANT ------
plot_zero_c = True
plot_emp_c = True
plot_svgd_c = True
plot_gif = False

n_particles = 1
time_plot = [13, 24, 100]

col_av = True
obstacle = True
is_linear = False
t_end = 100
std_ini = 0.2
n_agents = 2
num_rollouts = 30
# ------------------------

exp_name = 'robotsX'
exp_name += '_col_av' if col_av else ''
exp_name += '_obstacle' if obstacle else ''
exp_name += '_lin' if is_linear else '_nonlin'


# ------------ 0. Load ------------
# load data
f_data = 'data_T' + str(t_end) + '_stdini' + str(std_ini) + '_agents' + str(n_agents) + '_RS' + str(random_seed)
f_data += '.pkl'
filename_data = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', f_data)
filehandler = open(filename_data, 'rb')
data_saved = pickle.load(filehandler)
filehandler.close()
x0 = data_saved['x0'].to(device)
xbar = data_saved['xbar'].to(device)

# ------------ 1. Dataset ------------
train_data = data_saved['train_data_full'][:num_rollouts, :, :]
assert train_data.shape[0] == num_rollouts
test_data = data_saved['test_data']
# data for plot
t_ext = t_end * 4
plot_data = torch.zeros(t_ext, train_data.shape[-1])
plot_data[0, :] = (x0.detach() - xbar)
plot_data = plot_data.to(device)

n_test_trajectories = 12
plot_data_test = torch.zeros(n_test_trajectories, t_ext, test_data.shape[-1])
for i in range(n_test_trajectories):
    plot_data_test[i,0,:] = test_data[i,0,:]
train_points = (train_data + xbar)[:,0,:]

# ------------ 2. Parameters and hyperparameters ------------
# define the model
k = 1.0         # spring constant
u_init = None   # all zero
x_init = None   # same as xbar
sys = SystemRobots(
    xbar=xbar, x_init=x_init, u_init=u_init, is_linear=is_linear, k=k
)

# ------------ 3. Load models and defune controllers ------------

# define a zero controller
if plot_zero_c:
    ctl_zero = RENController(
        sys.noiseless_forward, num_states=sys.num_states,
        num_inputs=sys.num_inputs, output_amplification=0,
        n_xi=1, l=1, x_init=sys.x_init, u_init=sys.u_init,
        initialization_std=0, train_method='empirical',
    )

model_keys = ['X_vec', 'Y_vec', 'B2_vec', 'C2_vec', 'D21_vec', 'D22_vec', 'D12_vec']

# define the empirical controller
if plot_emp_c:
    # Load saved model for the empirical controller
    f_model = exp_name + '_emp_T' + str(t_end) + '_S' + str(num_rollouts) + '_stdini' + str(std_ini)
    f_model += '_agents' + str(n_agents) + '_RS' + str(random_seed) + '.pt'
    print('Loading ' + f_model)
    filename_model = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', 'trained_models', f_model)
    res_dict = torch.load(filename_model)
    assert num_rollouts == res_dict['num_rollouts']
    n_xi, l = res_dict['n_xi'], res_dict['l']
    initialization_std = res_dict['initialization_std']
    ctl_emp = RENController(
        sys.noiseless_forward, num_states=sys.num_states,
        num_inputs=sys.num_inputs, output_amplification=20,
        n_xi=n_xi, l=l, x_init=sys.x_init, u_init=sys.u_init,
        initialization_std=initialization_std, train_method='empirical',
    )
    for model_key in model_keys:
        ctl_emp.set_parameter(model_key, res_dict[model_key])
    ctl_emp.psi_u.eval()

# define the SVGD controller
if plot_svgd_c:
    ctl_svgd = [None]*n_particles
    for p in range(n_particles):
        # Load saved model
        f_model = exp_name + '_SVGDparticle' + str(p) + '_T' + str(t_end) + '_S' + str(num_rollouts)
        f_model += '_stdini' + str(std_ini) + '_agents' + str(n_agents) + '_RS'+str(random_seed) + '.pt'
        print('Loading ' + f_model)
        filename_model = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', 'trained_models', f_model)
        res_dict_particle = torch.load(filename_model, map_location=torch.device(device))
        n_xi, l = res_dict_particle['n_xi'], res_dict_particle['l']
        ctl_svgd[p] = RENController(
            sys.noiseless_forward, num_states=sys.num_states,
            num_inputs=sys.num_inputs, output_amplification=20,
            n_xi=n_xi, l=l, x_init=sys.x_init, u_init=sys.u_init,
            initialization_std=res_dict_particle['initialization_std'],
            train_method='SVGD',
        )
        # Set state dict
        for model_key in model_keys:
            ctl_svgd[p].set_parameter(model_key, res_dict_particle[model_key].to(device))
        ctl_svgd[p].psi_u.eval()

# ------------ 4. Plots ------------

# Simulate trajectory for zero controller
if plot_zero_c:
    # x_zero, _, u_zero = sys.rollout(ctl_zero, plot_data)
    x_zero1, _, u_zero = sys.rollout(ctl_zero, plot_data_test[0])
    x_zero2, _, _ = sys.rollout(ctl_zero, plot_data_test[1])
    x_zero3, _, _ = sys.rollout(ctl_zero, plot_data_test[2])
    # print(torch.abs(u_zero).sum())
    # plot trajectory
    tp = 22
    plot_trajectories(x_zero1, xbar, n_agents, exp_name=exp_name, obst=True,  circles=False, save=False, T=0)
    plot_trajectories(x_zero2, xbar, n_agents, exp_name=exp_name, obst=False, circles=False, save=False, T=0)
    plot_trajectories(x_zero3, xbar, n_agents, exp_name=exp_name, obst=False, circles=True,  save=False, T=tp)
    # adjust the figure
    fig = plt.gcf()
    fig.set_size_inches(6,6)
    plt.axis('equal')
    plt.tight_layout()
    ax = plt.gca()
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "serif"
    plt.text(0., 2.9, r'Open-loop system', dict(size=25), ha='center', va='top')
    plt.text(0., -2.9, r'$(a)$', dict(size=25), ha='center')
    plt.text(2.9, -2.9, r'$\tau = %d$' % tp, dict(size=25), ha='right')
    # save figure
    f_figure = 'zero_controller_tp'+str(tp)+'.pdf'
    filename_figure = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', f_figure)
    plt.savefig(filename_figure, format='pdf')
    plt.close()

# Simulate trajectories for empirical controller
if plot_emp_c:
    x_emp1, _, u_emp1 = sys.rollout(ctl_emp, plot_data_test[0])
    x_emp2, _, u_emp2 = sys.rollout(ctl_emp, plot_data_test[1])
    x_emp3, _, u_emp3 = sys.rollout(ctl_emp, plot_data_test[2])
    for idx,tp in enumerate(time_plot):
        # plot trajectories
        plot_trajectories(x_emp1, xbar, n_agents, exp_name='', obst=True,  circles=False, save=False, T=0)
        plot_trajectories(x_emp2, xbar, n_agents, exp_name='', obst=False, circles=False, save=False, T=0)
        plot_trajectories(x_emp3, xbar, n_agents, exp_name='', obst=False, circles=True,  save=False, T=tp)
        # plot points of initial conditions
        plt.plot(train_points[:30, 0], train_points[:30, 1], 'o', color='tab:blue', alpha=0.1)
        plt.plot(train_points[:30, 4], train_points[:30, 5], 'o', color='tab:orange', alpha=0.1)
        # adjust the figure
        fig = plt.gcf()
        fig.set_size_inches(6,6)
        plt.axis('equal')
        plt.tight_layout()
        text = r'$(b)$' if idx == 0 else r'$(c)$'
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = "serif"
        plt.text(0., 2.9, r'Empirical controller', dict(size=25), ha='center', va='top')
        plt.text(0., -2.9, text, dict(size=25), ha='center')
        plt.text(2.9, -2.9, r'$\tau = %d$' % tp, dict(size=25), ha='right')
        # save figure
        f_figure = 'xy_trajectories_after_train'
        f_figure += exp_name+'_emp_T'+str(t_end)+'_S'+str(num_rollouts)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)
        f_figure += 'screenshot_time'+str(tp)+'.pdf'
        filename_figure = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', f_figure)
        plt.savefig(filename_figure, format='pdf')
        plt.close()

# Simulate trajectories for svgd controller
if plot_svgd_c:
    for p in range(n_particles):
        x_svgd1, _, _ = sys.rollout(ctl_svgd[p], plot_data_test[0])
        x_svgd2, _, _ = sys.rollout(ctl_svgd[p], plot_data_test[1])
        x_svgd3, _, _ = sys.rollout(ctl_svgd[p], plot_data_test[2])
        for idx,tp in enumerate(time_plot):
            # plot trajectories
            plot_trajectories(x_svgd1, xbar, n_agents, exp_name='', obst=True,  circles=False, save=False, T=0)
            plot_trajectories(x_svgd2, xbar, n_agents, exp_name='', obst=False, circles=False, save=False, T=0)
            plot_trajectories(x_svgd3, xbar, n_agents, exp_name='', obst=False, circles=True,  save=False, T=tp)
            # plot points of initial conditions
            plt.plot(train_points[:, 0], train_points[:, 1], 'o', color='tab:blue', alpha=0.1)
            plt.plot(train_points[:, 4], train_points[:, 5], 'o', color='tab:orange', alpha=0.1)
            # adjust the figure
            fig = plt.gcf()
            fig.set_size_inches(6,6)
            plt.axis('equal')
            plt.tight_layout()
            text = r'$(d)$' if idx == 0 else (r'$(e)$' if idx == 1 else r'$(f)$')
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = "serif"
            plt.text(0., 2.9, r'Our controller', dict(size=25), ha='center', va='top')
            plt.text(0., -2.9, text, dict(size=25), ha='center')
            plt.text(2.9, -2.9, r'$\tau = %d$' % tp, dict(size=25), ha='right')
            # save figure
            f_figure = 'xy_trajectories_after_train' + exp_name + '_particle' + str(p)
            f_figure += '_T'+str(t_end)+'_S'+str(num_rollouts)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)
            f_figure += 'screenshot_time'+str(tp)+'.pdf'
            filename_figure = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results', f_figure)
            plt.savefig(filename_figure, format='pdf')
            plt.close()

# ------------ 5. GIFs ------------

# Base controller
if plot_zero_c and plot_gif:
    for idx, x in enumerate([x_zero1,x_zero2,x_zero3]):
        print("Generating figures for OL trajectory %d..." % (idx+1))
        for tp in range(1, t_end):
            ob_print = True
            if idx > 0:
                plot_trajectories(x_zero1, xbar, n_agents, exp_name='', obst=ob_print, save=False, T=1)
                ob_print = False
            if idx == 2:
                plot_trajectories(x_zero2, xbar, n_agents, exp_name='', save=False, T=1)
            plot_trajectories(x, xbar, n_agents, exp_name='', obst=ob_print, circles=True, axis=True, save=False, T=tp)
            # plot points of initial conditions
            plt.plot(train_points[:, 0], train_points[:, 1], 'o', color='tab:blue', alpha=0.1)
            plt.plot(train_points[:, 4], train_points[:, 5], 'o', color='tab:orange', alpha=0.1)
            # adjust the figure
            fig = plt.gcf()
            fig.set_size_inches(6, 6)
            plt.axis('equal')
            plt.tight_layout()
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = "serif"
            plt.text(0., 2.9, r'Open-loop system', dict(size=25), ha='center', va='top')
            plt.text(1.85, -2.9, r'$\tau = %02d$' % tp, dict(size=25), ha='left')
            f_gif = "ol_%03i" % (tp + 100*idx) + ".png"
            filename_gif = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'gif', f_gif)
            plt.savefig(filename_gif)
            plt.close(fig)
    print("Generating OL gif...")
    filename_figs = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'gif', "ol_*.png")
    filename_gif = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'gif', "ol.gif")
    command = "convert -delay 4 -loop 0 " + filename_figs + " " + filename_gif
    os.system(command)
    print("Gif saved at %s" % filename_gif)
    print("Deleting figures...")
    command = "rm " + filename_figs
    os.system(command)

# Empirical controller
if plot_emp_c and plot_gif:
    for idx, x in enumerate([x_emp1,x_emp2,x_emp3]):
        print("Generating figures for emp trajectory %d..." % (idx+1))
        for tp in range(1, t_end):
            ob_print = True
            if idx > 0:
                plot_trajectories(x_emp1, xbar, n_agents, exp_name='', obst=ob_print, save=False, T=1)
                ob_print = False
            if idx == 2:
                plot_trajectories(x_emp2, xbar, n_agents, exp_name='', save=False, T=1)
            plot_trajectories(x, xbar, n_agents, exp_name='', obst=ob_print, circles=True, axis=True, save=False, T=tp)
            # plot points of initial conditions
            plt.plot(train_points[:, 0], train_points[:, 1], 'o', color='tab:blue', alpha=0.1)
            plt.plot(train_points[:, 4], train_points[:, 5], 'o', color='tab:orange', alpha=0.1)
            # adjust the figure
            fig = plt.gcf()
            fig.set_size_inches(6, 6)
            plt.axis('equal')
            plt.tight_layout()
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = "serif"
            plt.text(0., 2.9, r'Empirical controller', dict(size=25), ha='center', va='top')
            plt.text(1.85, -2.9, r'$\tau = %02d$' % tp, dict(size=25), ha='left')
            f_gif = "emp_%03i" % (tp + 100*idx) + ".png"
            filename_gif = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'gif', f_gif)
            plt.savefig(filename_gif)
            plt.close(fig)
    print("Generating emp gif...")
    filename_figs = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'gif', "emp_*.png")
    filename_gif = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'gif', "emp.gif")
    command = "convert -delay 4 -loop 0 " + filename_figs + " " + filename_gif
    os.system(command)
    print("Gif saved at %s" % filename_gif)
    print("Deleting figures...")
    command = "rm " + filename_figs
    os.system(command)

# SVGD controller
if plot_svgd_c and plot_gif:
    if n_particles != 1:
        print("Gif will be generated for particle %i" % (n_particles-1))
    for idx, x in enumerate([x_svgd1,x_svgd2,x_svgd3]):
        print("Generating figures for svgd trajectory %d..." % (idx+1))
        for tp in range(1, t_end):
            ob_print = True
            if idx > 0:
                plot_trajectories(x_svgd1, xbar, n_agents, exp_name='', obst=ob_print, save=False, T=1)
                ob_print = False
            if idx == 2:
                plot_trajectories(x_svgd2, xbar, n_agents, exp_name='', save=False, T=1)
            plot_trajectories(x, xbar, n_agents, exp_name='', obst=ob_print, circles=True, axis=True, save=False, T=tp)
            # plot points of initial conditions
            plt.plot(train_points[:, 0], train_points[:, 1], 'o', color='tab:blue', alpha=0.1)
            plt.plot(train_points[:, 4], train_points[:, 5], 'o', color='tab:orange', alpha=0.1)
            # adjust the figure
            fig = plt.gcf()
            fig.set_size_inches(6, 6)
            plt.axis('equal')
            plt.tight_layout()
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = "serif"
            plt.text(0., 2.9, r'Our controller', dict(size=25), ha='center', va='top')
            plt.text(1.85, -2.9, r'$\tau = %02d$' % tp, dict(size=25), ha='left')
            f_gif = "svgd_%03i" % (tp + 100*idx) + ".png"
            filename_gif = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'gif', f_gif)
            plt.savefig(filename_gif)
            plt.close(fig)
    print("Generating svgd gif...")
    filename_figs = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'gif', "svgd_*.png")
    filename_gif = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'gif', "svgd.gif")
    command = "convert -delay 4 -loop 0 " + filename_figs + " " + filename_gif
    os.system(command)
    print("Gif saved at %s" % filename_gif)
    print("Deleting figures...")
    command = "rm " + filename_figs
    os.system(command)
