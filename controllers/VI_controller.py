import torch, time, copy
import numpy as np
from numpy import random
from config import device
from controllers.abstract import get_controller
from assistive_functions import WrapLogger, DummyLRScheduler, to_tensor
from inference_algs.distributions import GibbsPosterior


class VICont():
    def __init__(
        self, sys, train_d, lr, loss, prior_dict, controller_type,
        random_seed=None, optimizer='Adam', batch_size=-1, lambda_=None,
        num_iter_fit=None, lr_decay=1.0, logger=None,
        # VI properties
        num_vfs=10, vf_init_std=0.1, vf_cov_type='diag', vf_param_dists=None, L=1,
        # REN controller properties
        n_xi=None, l=None, x_init=None, u_init=None,
        # debug
        debug=False, mu_debug=5, var_debug=0.5,
    ):
        """
        train_d: (ndarray) train disturbances - shape: (S, T, sys.num_states)
        L: number of samples from the posterior to approximate ELBO
        """
        # --- init ---
        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            self.random_state = random.RandomState(random_seed)
        else:
            self.random_state = random.RandomState(0)

        self.train_d = to_tensor(train_d)
        self.num_iter_fit = num_iter_fit
        self.best_vfs = None
        if batch_size < 1:
            self.batch_size = self.train_d.shape[0]
        else:
            self.batch_size = min(batch_size, self.train_d.shape[0])

        (S, T, num_states) = self.train_d.shape
        assert num_states == sys.num_states
        assert T == loss.T
        assert optimizer in ['Adam', 'SGD']
        assert controller_type in ['REN', 'Affine']

        self.fitted, self.over_fitted = False, False
        self.unknown_err = False
        self.logger = WrapLogger(logger)

        self.debug = debug
        if self.debug:
            self.logger.info(
                '[INFO] Debug mode. Posteior is N('+str(mu_debug)+', '+str(var_debug)+'), not Gibbs.'
            )
            self.mu_debug, self.var_debug = mu_debug, var_debug

        """ --- Setup model & inference --- """
        self._setup_model_inference(
            sys=sys, lambda_=lambda_, loss=loss, prior_dict=prior_dict,
            optimizer=optimizer, lr=lr,
            lr_decay=lr_decay,
            controller_type=controller_type,
            n_xi=n_xi, l=l, x_init=x_init, u_init=u_init,
            # VI properties
            num_vfs=num_vfs, vf_init_std=vf_init_std, L=L,
            vf_cov_type=vf_cov_type, vf_param_dists=vf_param_dists,
        )

        self.fitted = False


    # -------- FIT --------
    def fit(self, early_stopping=True, log_period=500):
        """
        fits the variational posterior

        Args:
            early_stopping: return model at an evaluated iteration with the lowest VI loss
            log_period (int) number of steps after which to print stats
        """

        self.best_vfs = None
        min_criterion = 1e6

        t = time.time()
        itr = 1
        loss_VI_history = [None]*log_period
        # vf_history = [None]*log_period
        while itr <= self.num_iter_fit:
            sel_inds = self.random_state.choice(self.train_d.shape[0], size=self.batch_size)
            task_dict_batch = self.train_d[sel_inds, :, :]
            # --- take a step ---
            self.optimizer.zero_grad()

            # try:
            loss_VI = self.get_neg_elbo(task_dict_batch)
            loss_VI.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # add loss VI to history
            loss_VI_history = loss_VI_history[1:]     # drop oldest
            loss_VI_history.append(loss_VI.item())    # add newest
            # vf_history = vf_history[1:]         # drop oldest
            # vf_history.append((
            #     self.var_post.loc.detach().cpu().numpy()[1],
            #     self.var_post.scale_raw.detach().cpu().numpy()[1],
            #     self.var_post.loc.detach().cpu().numpy()[0],
            #     self.var_post.scale_raw.detach().cpu().numpy()[0]
            # ))    # add newest

            # --- print stats ---
            if (itr % log_period == 0) and (not self.unknown_err):
                # average scale
                scale_av = torch.mean(torch.exp(self.var_post.scale_raw.flatten()))

                # evaluate mean of cuurent variational factor
                loss = self.eval(controller_params=self.var_post.loc, data=task_dict_batch)

                # log info
                duration = time.time() - t
                t = time.time()
                message = '\nIter %d/%d - Time %.2f sec - VI Loss %.4f - Av. VI Loss %.4f - Loss %.4f - Scale Av %.4f' % (
                    itr, self.num_iter_fit, duration, loss_VI.item(), sum(loss_VI_history)/log_period, loss, scale_av
                )
                # log learning rate
                # message += ', LR: '+str(self.lr_scheduler.get_last_lr())
                self.logger.info(message)

                # update the best VFs based on average VI loss
                if sum(loss_VI_history)/log_period < min_criterion:
                    min_criterion = sum(loss_VI_history)/log_period
                    self.best_vfs = copy.deepcopy(self.var_post)
                    self.logger.info('updated best variational factors.')

            # stop training
            if self.over_fitted or self.unknown_err:
                break

            # update learning rate
            self.lr_scheduler.step()
            # go to next iter
            itr = itr+1

        self.fitted = True if not self.unknown_err else False

        # set back to the best VFs if early stopping
        if early_stopping and (not self.best_vfs is None):
            self.var_post = self.best_vfs



    # -------------------------
    # -- negative ELBO loss ---
    # -------------------------
    def get_neg_elbo(self, tasks_dicts):
        '''
        notation: true dist (P), latent var (Z), observed (X), var dist (Q)
        ELBO = E_Q [log P(X, Z) - loq q(z)]
        P(X, Z) is the numerator of the unnormalized true posterior
        expecation is estimated by sample mean over self.L sampled params from Q.
        '''
        param_sample = self.var_post.rsample(sample_shape=(self.L,))
        if self.debug:
            if self.num_vfs>1:
                raise NotImplementedError
            log_prob_normal_debug = -0.5/self.var_debug * torch.matmul(
                param_sample-self.mu_debug,
                torch.transpose(param_sample-self.mu_debug, 0, 1)
            )
            elbo = log_prob_normal_debug - self.var_post.log_prob(param_sample)
        else:
            # tile data to number or VFs
            # data_tuples_tiled = _tile_data_tuples(tasks_dicts, self.num_vfs)
            data_tuples_tiled = tasks_dicts #TODO: use the above line
            log_posterior_num = self.generic_Gibbs.log_prob(param_sample, data_tuples_tiled)    # log
            log_var_post = self.var_post.log_prob(param_sample)
            # print('-log_posterior_num', -torch.mean(log_posterior_num), '-log_var_post', -torch.mean(log_var_post))
            elbo = log_posterior_num - log_var_post
        elbo = elbo.reshape(self.L)
        assert elbo.ndim == 1 and elbo.shape[0] == self.L
        return - torch.mean(elbo)

    # -------------------------
    # ------ setup model ------
    def _setup_model_inference(
        self, sys, lambda_, loss, prior_dict,
        optimizer, lr, lr_decay,
        controller_type, n_xi, l, x_init, u_init,
        # VI properties
        num_vfs, vf_init_std, vf_cov_type, vf_param_dists, L
    ):
        self.num_vfs, self.L = num_vfs, L

        """define a generic controller"""
        # define a generic controller
        self.generic_controller = get_controller(
            controller_type=controller_type, sys=sys,
            # REN
            n_xi=n_xi, l=l, x_init=x_init, u_init=u_init,
            initialization_std=0.1, # for initializing REN. not important
        )

        """define a generic Gibbs posterior"""
        # only used to know how many parameters we have
        self.generic_Gibbs = GibbsPosterior(
            controller=self.generic_controller,
            sys=copy.deepcopy(sys), loss_fn=loss,
            lambda_=lambda_, prior_dict=prior_dict,
            logger=self.logger
        )

        """ variational posterior """
        self.var_post = GaussVarPosterior(
            named_param_shapes=self.generic_Gibbs.parameter_shapes(),
            vf_init_std=vf_init_std, vf_cov_type=vf_cov_type,
            vf_param_dists=vf_param_dists
        )

        """ optimizer """
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.var_post.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.var_post.parameters(), lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

    # ------------------------

    def eval(self, controller_params, data, loss_fn=None):
        """
        evaluates a controller with parameters 'controller_params' on 'data'.
        if 'loss_fn' is None, uses the bounded loss function as in Gibbs posterior.
        """
        # use default loss function if not specified
        loss_fn = loss_fn if not loss_fn is None else self.generic_Gibbs.loss_fn
        # Set state dict
        self.generic_controller.reset()
        self.generic_controller.set_parameters_as_vector(controller_params)
        self.generic_controller.psi_u.eval()
        with torch.no_grad():
            # rollout
            xs, _, us = self.generic_Gibbs.generic_cl_system.sys.rollout(self.generic_controller, data)
            # loss
            loss = loss_fn.forward(xs, us)
        return loss.item()

    # def _vectorize_pred_dist(self, pred_dist):
    #     multiv_normal_batched = pred_dist.dists
    #     normal_batched = torch.distributions.Normal(multiv_normal_batched.mean, multiv_normal_batched.stddev)
    #     return EqualWeightedMixtureDist(normal_batched, batched=True, num_dists=multiv_normal_batched.batch_shape[0])


# ---------------------------------------
import math
from collections import OrderedDict
from pyro.distributions import MultivariateNormal, Normal

class GaussVarPosterior(torch.nn.Module):
    '''
    Gaussian Variational posterior on the controller parameters
    '''

    def __init__(self, named_param_shapes, vf_init_std=0.1, vf_cov_type='full', vf_param_dists=None):
        '''
        - named_param_shapes: list of (name, shape) of controller's parameters
        - vf_init_std: for each Variational Factor (VF),
                        a) every element of its mean is initialized from N(0, vf_init_std).
                        b) if vf_cov_type=='diag', every element of the diagonal of cov is
                           initialized from N(ln(0.1), vf_init_std).
                        c) if vf_cov_type=='full', vf_init_std is not used. every element in the
                           lower triangle of cov is initialized from U(0.05, 0.1).
        - vf_cov_type: type of covariance for each Gaussian VF.
        - vf_param_dists: distribution to initialize VFs from. If None, do as above.
        '''
        super().__init__()

        self.vf_cov_type = vf_cov_type
        assert vf_cov_type in ['diag', 'full']

        # --- define shapes ---
        self.param_idx_ranges = OrderedDict()
        idx_start = 0
        for name, shape in named_param_shapes.items():
            assert len(shape) == 1
            idx_end = idx_start + shape[0]
            self.param_idx_ranges[name] = (idx_start, idx_end)
            idx_start = idx_end
        param_shape = torch.Size((idx_start,))

        # --- initialize VI ---
        if vf_param_dists is None: # initialize randomly
            self.loc = torch.nn.Parameter(
                torch.normal(0.0, vf_init_std, size=param_shape, device=device))
            if self.vf_cov_type == 'diag':
                self.scale_raw = torch.nn.Parameter(
                    torch.normal(math.log(0.1), vf_init_std, size=param_shape, device=device)
                )
                self.dist_fn = lambda: Normal(
                    self.loc,
                    self.scale_raw.exp()
                ).to_event(1)
            elif self.vf_cov_type == 'full':
                self.scale_raw=torch.nn.Parameter(
                    torch.normal(
                        math.log(0.1), vf_init_std,
                        size=param_shape[0]*(param_shape[0]+1)/2, device=device
                    )
                )
                tril_scale_raw = torch.tril(torch.exp(self.scale_raw))
                covariance_matrix = torch.matmul(torch.transpose(tril_scale_raw), tril_scale_raw)
                assert covariance_matrix.shape == (param_shape[0], param_shape[0])
                self.dist_fn = lambda: MultivariateNormal(
                        loc=self.loc,
                        covariance_matrix=covariance_matrix
                    ) #.to_event(1)
            else:
                raise NotImplementedError
        else: # initialize from given vf_param_dists
            loc=[]
            scale_raw = []
            for name, shape in named_param_shapes.items():
                loc.append(vf_param_dists[name]['mean'])
                if 'variance' in vf_param_dists[name].keys():
                    scale_raw.append(math.log(math.sqrt(vf_param_dists[name]['variance'])))
                elif 'scale' in vf_param_dists[name].keys():
                    scale_raw.append(math.log(vf_param_dists[name]['scale']))
                elif 'scale_raw' in vf_param_dists[name].keys():
                    scale_raw.append(vf_param_dists[name]['scale_raw'])
                else:
                    raise NotImplementedError
            self.loc = torch.nn.Parameter(torch.tensor(loc).float().to(device))
            self.scale_raw = torch.nn.Parameter(torch.tensor(scale_raw).float().to(device))
            if self.vf_cov_type == 'diag':
                assert self.scale_raw.shape==(param_shape)
                self.dist_fn = lambda: Normal(
                        self.loc,
                        self.scale_raw.exp()
                    ).to_event(1)
            elif self.vf_cov_type == 'full':
                assert self.scale_raw.shape==(param_shape[0]*(param_shape[0]+1)/2)
                tril_scale_raw = torch.tril(torch.exp(self.tril_scale_raw))
                covariance_matrix = torch.matmul(torch.transpose(tril_scale_raw), tril_scale_raw)
                assert covariance_matrix.shape == (param_shape[0], param_shape[0])
                self.dist_fn = lambda: MultivariateNormal(
                        loc=self.loc,
                        covariance_matrix=covariance_matrix
                    ) #.to_event(1)
            else:
                raise NotImplementedError



    def forward(self):
        return self.dist_fn()

    def rsample(self, sample_shape=torch.Size()):
        return self.forward().rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()):
        return self.forward().sample(sample_shape)

    def log_prob(self, value):
        return self.forward().log_prob(value)

    def parameters_dict(self):
        return {'loc':self.loc.detach().clone(), 'scale_raw':self.scale_raw.detach().clone()}

    @property
    def mode(self):
        return self.mean

    @property
    def mean(self):
        return self.forward().mean

    @property
    def stddev(self):
        return self.forward().stddev

    def entropy(self):
        return self.forward().entropy()

    @property
    def mean_stddev_dict(self):
        # dictionaey of mean and stddev for each group of parameters
        mean = self.mean
        stddev = self.stddev
        with torch.no_grad():
            return OrderedDict(
                [(name, (mean[idx_start:idx_end], stddev[idx_start:idx_end])) for name, (idx_start, idx_end) in self.param_idx_ranges.items()])




def _tile_data_tuples(tasks_dicts, tile_size):
    train_data_tuples_tiled = []
    for task_dict in tasks_dicts:
        x_data = task_dict
        x_data = x_data.view(torch.Size((1,)) + x_data.shape).repeat(tile_size, 1, 1)
        train_data_tuples_tiled.append((x_data))
    return train_data_tuples_tiled