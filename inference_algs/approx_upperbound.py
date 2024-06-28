import torch
import numpy as np

from controllers.abstract import AffineController


def approx_Z(grid_dict, sys, lq_loss, data, lambda_):
    # grid_dict has fields theta, bias, prior
    assert set(grid_dict.keys()) == set(['bias', 'theta', 'prior'])
    if not abs(sum(grid_dict['prior'])-1)<1e-6:
        print(
            '[WARN] prior has a support further than the provided grid \
            (integral in the grid = {:1.4f}). This results in an approximation \
            lower than the upper bound.'.format(sum(grid_dict['prior']))
        )
    res = 0
    for ind in range(len(grid_dict['theta'])):
        # set params
        controller_tmp = AffineController(
            weight=np.array([[grid_dict['theta'][ind]]]),
            bias=np.array([[grid_dict['bias'][ind]]])
        )
        # roll out
        xs_train, _, us_train = sys.rollout(
            controller_tmp,
            data
        )
        assert xs_train.shape == data.shape
        # compute LQR loss
        with torch.no_grad():
            loss = lq_loss.forward(xs_train, us_train).item()
        #
        res += grid_dict['prior'][ind] * np.exp(-lambda_*loss)

    return res


def approx_upper_bound(
    grid_dict, sys, lq_loss, data, lambda_,
    epsilon, loss_bound, approximated_Z=None
):
    """
    approximates the upper bound by gridding over the prior.
    approximated_Z is the denumenator of the posterior, which is used to
    normalized the posterior when computing it. If this value is saved,
    it can be provided to the function. otherwise, is recomputed.
    """
    assert len(data.shape) == 3, 'data must be of shape (S, T, num_states)'
    num_rollouts = data.shape[0]
    if approximated_Z is None:
        print('Approximating Z.')
        approximated_Z = approx_Z(grid_dict, sys, lq_loss, data, lambda_)
    else:
        print('Approximating Z provided to the function.')
    print('lambda C^2 / 8 s', lambda_*(loss_bound**2)/8/num_rollouts)
    print('- 1/lambda ln(Z)', - 1/lambda_*np.log(approximated_Z))
    print('1/lambda ln(1/epsilon)', 1/lambda_*np.log(1/epsilon))

    return lambda_*(loss_bound**2)/8/num_rollouts - 1/lambda_*np.log(approximated_Z) + 1/lambda_*np.log(1/epsilon)


def approx_Z_sample_base(
    prior_dist, d, n_p, data_train, ctl_generic, sys, lq_loss_bounded, lambda_, delta_hat,
    adaptive_np, max_np
):
    """
    Approximates Z_lambda using the sample-based method in Section V-A.
    inputs:
        - prior_dist: prior distribution with 'sample' function.
        - d: dimension of the controller parameters
        - n_p: number of sampled controllers from the prior
        - data_train:
        - ctl_generic: generic controller. note that the params in prior loc and in the
                       controller must be in the same order.
        - sys
        - lq_loss_bounded: bounded loss function
        - lambda_
    """

    # assertions
    if adaptive_np:
        assert not max_np is None
        assert n_p <= max_np

    C = lq_loss_bounded.loss_bound
    res = 0
    n_p_old = 0
    while True:
        # 1. sum loss over samples from the prior
        for i in range(n_p-n_p_old):
            # sample theta_i from the prior
            theta_i = prior_dist.sample()
            # define controller
            ctl_generic.set_vector_as_params(theta_i)
            # roll
            x_tmp, _, u_tmp = sys.rollout(ctl_generic, data_train)
            # loss
            train_loss_bounded = lq_loss_bounded.forward(x_tmp, u_tmp).item()
            res += torch.exp(- lambda_ * train_loss_bounded)

        # approx_Z = res/n_p - const
        const = (1-torch.exp(-lambda_*C))*(torch.log(1/delta_hat)/2/n_p)**0.5

        # exit conditions: not adaptive or results in a positive Z or max n_p reached
        if (not adaptive_np) or (res/n_p>const) or (n_p >= max_np):
            break

        # increase n_p
        n_p_old = n_p
        n_p += 100
        print('[INFO] increased n_p to {:.0f} since approximated Z was not positive.'.format(n_p))

    # out of the loop
    approx_Z = res/n_p - const
    # final check
    if approx_Z <=0:
        print('[ERR] approximated Z is negative. consider increasing n_p or delta_hat.')
        # raise ValueError(msg)

    return approx_Z, n_p


def approx_ub_sample_base(
    epsilon,
    # for approximating Z
    prior_dist, d, n_p, data_train, ctl_generic, sys, lq_loss_bounded, lambda_, delta_hat,
    adaptive_np=False, max_np=None
):
    """
    Approximates the upper bound using the sample-based method in Section V-A.
    inputs:
        - epsilon: confidence level for PAC
        - prior_dist: prior distribution with 'sample' function
        - d: dimension of the controller parameters
        - n_p: number of sampled controllers from the prior
        - data_train:
        - ctl_generic: generic controller. note that the params in prior loc and in the
                       controller must be in the same order.
        - sys
        - lq_loss_bounded: bounded loss function
        - lambda_
        - adaptive_n_p: if n_p was too small that approximated Z was negative, increase it
        - max_np: if adaptive_n_p, increase n_p until this
    """

    # init
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = torch.tensor(lambda_)
    if not isinstance(delta_hat, torch.Tensor):
        delta_hat = torch.tensor(delta_hat)
    C = lq_loss_bounded.loss_bound
    num_rollouts = data_train.shape[0]

    # initial warning
    if 2*n_p/torch.log(1/delta_hat) <= (torch.exp(lambda_*C)-1)**2:
        print('[WARN] approximated Z <might become> negative. consider increasing n_p or delta_hat.')

    approximarted_Z_sample_base, used_np = approx_Z_sample_base(
        prior_dist=prior_dist, d=d, n_p=n_p, data_train=data_train,
        ctl_generic=ctl_generic, sys=sys, lq_loss_bounded=lq_loss_bounded,
        lambda_=lambda_, delta_hat=delta_hat, adaptive_np=adaptive_np, max_np=max_np
    )

    return lambda_*(C**2)/8/num_rollouts - 1/lambda_*np.log(approximarted_Z_sample_base) + 1/lambda_*np.log(1/epsilon), used_np
