import os, pickle, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)


# ------ LOAD DATA ------
def load_data(dist_type, S, T, random_seed, S_test = None):
    '''
    use a subset of available test data if S_test is not None
    '''
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
    data_train = data_all['train_big'][dist_type][:S, :, :] if not S is None else data_all['train_big'][dist_type]
    if not S_test is None:
        data_test = data_all['test_big'][dist_type][:S_test, :, :]
    else:
        data_test = data_all['test_big'][dist_type]
    # disturbance
    disturbance = data_all['disturbance']

    return data_train, data_test, disturbance


# ------ COMPUTE POSTERIOR BY GRIDDING ------
import itertools, math, torch
from controllers.abstract import AffineController

def compute_posterior_by_gridding(
    prior_dict, lq_loss_bounded, data_train,
    dist_type, sys, gibbs_lambda, n_grid
):
    S, T, _ = data_train.shape

    prior_type_b = prior_dict['type_b']
    prior_type_w = prior_dict['type_w']

    # ------ grid ------
    if prior_type_w == 'Uniform':
        theta_grid = torch.linspace(
            prior_dict['weight_low'], prior_dict['weight_high'], n_grid
        )
    elif prior_type_w == 'Gaussian':
        theta_grid = torch.linspace(
            prior_dict['weight_loc']-2,
            prior_dict['weight_loc']+2,
            n_grid
        )
    else:
        raise NotImplementedError

    if prior_type_b == 'Uniform':
        bias_grid = torch.linspace(
            prior_dict['bias_low'], prior_dict['bias_high'],
            n_grid
        )
    elif prior_type_b == 'Uniform_pos':
        bias_grid = torch.linspace(-5, 5, n_grid)
        # should consider the full range, b.c. prior is on the wrong side
    elif prior_type_b == 'Uniform_neg':
        n_grid = int((n_grid+1)/2)      # NOTE: range is half => half points. o.w., prior becomes half the full range
        bias_grid = torch.linspace(
            prior_dict['bias_low'], prior_dict['bias_high'],
            n_grid
        )
    elif prior_type_b == 'Gaussian':
        bias_grid = torch.linspace(-5, 5, n_grid)
    elif prior_type_b == 'Gaussian_biased_wide':
        bias_grid = torch.linspace(-5, 5, n_grid)
    elif prior_type_b == 'Gaussian_biased':
        n_grid = int((n_grid+1)/2)      # NOTE: range is half => half points. o.w., prior becomes half the full range
        bias_grid = torch.linspace(-5, 0, n_grid)
    else:
        raise NotImplementedError
    theta_grid, _ = torch.sort(theta_grid, descending=True)
    bias_grid, _ = torch.sort(bias_grid, descending=True)

    #  ------ prior ------
    if prior_type_w == 'Uniform':
        prior_w = 1/len(theta_grid)*torch.ones(len(theta_grid))
    elif prior_type_w == 'Gaussian':
        mean = prior_dict['weight_loc']
        sigma = prior_dict['weight_scale']
        prior_w = 1/sigma/(2*torch.tensor(math.pi))**0.5 * torch.exp(-(theta_grid-mean)**2/(2*sigma**2))
        prior_w = prior_w * abs(theta_grid[-1]-theta_grid[0])/len(theta_grid)
        prior_w = prior_w.flatten()
        # NOTE: convert continuous pdf to discrete histogram
        assert sum(prior_w) <= 1+1e-5, sum(prior_w)
    else:
        raise NotImplementedError
    if prior_type_b in ['Uniform', 'Uniform_neg']:
        prior_b = 1/len(bias_grid)*torch.ones(len(bias_grid))
    elif prior_type_b == 'Uniform_pos':
        prior_b = torch.cat((
            torch.zeros(int((len(bias_grid)-1)/2)),
            torch.ones(int((len(bias_grid)+1)/2))
        ))/int((len(bias_grid)+1)/2)
    elif prior_type_b.startswith('Gaussian'):
        mean = prior_dict['bias_loc']
        sigma = prior_dict['bias_scale']
        prior_b = 1/(sigma*torch.sqrt(2*torch.tensor(math.pi))) * torch.exp(-(bias_grid-mean)**2/(2*sigma**2))
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
        'posterior_unnorm':[None]*num_rows, 'posterior':[None]*num_rows,
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
        c_tmp = AffineController(
            torch.tensor([[theta_tmp]]), torch.tensor([[bias_tmp]])
        )
        # roll
        x_tmp, _, u_tmp = sys.rollout(
            c_tmp, data_train
        )
        # apply controller on train data
        train_loss_bounded = lq_loss_bounded.forward(x_tmp, u_tmp).item()
        # compute posterior unnormalized
        res_dict['posterior_unnorm'][ind] = res_dict['prior'][ind] * math.exp(
            -gibbs_lambda * train_loss_bounded
        )
    # NOTE: normalize
    sum_posterior = sum(res_dict['posterior_unnorm'])
    res_dict['posterior'] = [
        x/sum_posterior for x in res_dict['posterior_unnorm']
    ]
    approximated_Z = sum_posterior

    res_dict['theta_grid'] = theta_grid
    res_dict['bias_grid'] = bias_grid
    res_dict['approximated_Z'] = approximated_Z

    return res_dict


#
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl

def heatmap_dists(
    dists, theta_grid, bias_grid, extend_neg=True,
    titles=None, save_fig=False, marker_size=50
):
    if not titles is None:
        assert len(titles)==len(dists)
    vmin = 0
    vmax=max([np.max(np.max(dist)) for dist in dists])

    # ------ format ------
    plt.rcParams['text.usetex'] = True
    sns.set_theme(context='paper', style='white', palette='bright', font='sans-serif', font_scale=1.4, color_codes=True, rc=None)
    mpl.rc('font', family='serif', serif='Times New Roman')
    # ------

    fig, axs = plt.subplots(len(dists), 1,figsize=(4, 3*len(dists)))
    axs = [axs] if len(dists)==1 else axs

    X_ext = None
    X = -bias_grid
    Y = -theta_grid.flatten()

    for ind, dist in enumerate(dists):
        dist = np.reshape(
            np.array(dist),
            (len(theta_grid), len(bias_grid))
        )
        # assert sum(sum(dist))<=1+1e-5

        # extend to negative side
        if extend_neg and (X[0]*X[-1]>=0):
            if X_ext is None:
                d = np.mean(X[1:]-X[0:-1])
                stop = X[0] - d
                num = int(np.round((stop-(-5))/d))
                start = stop - num*d
                X_ext = np.concatenate((np.linspace(start, stop, num, endpoint=True), X))
            dist_ext = np.concatenate(
                (np.zeros((dist.shape[0], num)), dist), axis=1
            )
            assert dist_ext.shape==(len(Y), len(X_ext))
        else:
            X_ext, dist_ext = X, dist

        # ------ plot ------
        axs[ind].set_xlim([-5,5])
        im = axs[ind].imshow(
            dist_ext, origin='lower', extent=[X_ext[0], X_ext[-1], Y[0], Y[-1]],
            cmap='RdGy_r', alpha=0.5, aspect='auto', vmin=vmin, vmax=vmax, interpolation='bilinear')
        contours = axs[ind].contour(X, Y, dist, 4, colors='black', zorder=-1)
        axs[ind].clabel(contours, inline=True, fontsize=8)
        axs[ind].set_xlabel(r'$\beta$')
        axs[ind].set_ylabel(r'$k$')

        # set title
        if not titles is None:
            axs[ind].set_title(titles[ind])

        # # inside: loc='upper left'
        # axs[0,-1].legend(bbox_to_anchor=(1.9, 1.05))   # loc of upper-right corner of the legend. increasing moves it right and up
        # colorbar
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.82, 0.11, 0.03, 0.65])
        # fig.colorbar(im, cax=cbar_ax)

        # if save_fig:
        #     filename = dist_type.replace(" ", "_")+'_'+prior_type_b+'_contour_T'+str(T)+'_S'+str(S)+'.pdf'
        #     filename = os.path.join(file_path, filename)
        #     plt.savefig(filename)
    plt.tight_layout()
    plt.show()
