import torch, time, copy
import numpy as np
from numpy import random
from config import device
from assistive_functions import WrapLogger, DummyLRScheduler, to_tensor
from SVGD_src.distributions import GibbsPosterior


class VICont():
    def __init__(
        self, sys, train_d, lr, loss, prior_dict, controller_type,
        random_seed=None, optimizer='Adam', batch_size=-1, lambda_=None,
        num_iter_fit=None, lr_decay=1.0, logger=None,
        # VI properties
        num_vfs=10, vf_init_std=0.1, vf_cov_type='diag', vf_param_dists=None, L=1,
        # NN controller properties
        layer_sizes=None, nonlinearity_hidden=None, nonlinearity_output=None,
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
        assert controller_type in ['NN', 'REN', 'Affine']

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
            lr_decay=lr_decay, layer_sizes=layer_sizes,
            nonlinearity_hidden=nonlinearity_hidden,
            nonlinearity_output=nonlinearity_output,
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
            early_stopping: return model at an evaluated iteration with the lowest loss
            log_period (int) number of steps after which to print stats
        """

        self.best_vfs = None
        min_criterion = 1e6

        t = time.time()
        itr = 1
        loss_history = [None]*log_period
        vf_history = [None]*log_period
        while itr <= self.num_iter_fit:
            sel_inds = self.random_state.choice(self.train_d.shape[0], size=self.batch_size)
            task_dict_batch = self.train_d[sel_inds, :, :]
            # --- take a step ---
            self.optimizer.zero_grad()

            # try:
            loss = self.get_neg_elbo(task_dict_batch)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            # except Exception as e:
            #     self.logger.info('[Unhandled ERR] in VI step: ' + type(e).__name__ + '\n')
            #     self.logger.info(e)
            #     self.unknown_err = True

            # add loss to history
            loss_history = loss_history[1:]     # drop oldest
            loss_history.append(loss.item())    # add newest
            vf_history = vf_history[1:]         # drop oldest
            vf_history.append((
                self.var_post.loc.detach().cpu().numpy()[1],
                self.var_post.scale_raw.detach().cpu().numpy()[1],
                self.var_post.loc.detach().cpu().numpy()[0],
                self.var_post.scale_raw.detach().cpu().numpy()[0]
            ))    # add newest

            # --- print stats ---
            if (itr % log_period == 0) and (not self.unknown_err):
                duration = time.time() - t
                t = time.time()
                message = '\nIter %d/%d - Time %.2f sec - Loss %.4f - Av. Loss %.4f' % (
                    itr, self.num_iter_fit, duration, loss.item(), sum(loss_history)/log_period
                )
                # log info
                self.logger.info(message)
                # print('Current dists: weight N({:.2f}, {:.2f}), bias N({:.2f}, {:.2f})'.format(
                #     self.var_post.loc.detach().numpy()[1], np.exp(self.var_post.scale_raw.detach().numpy()[1]),
                #     self.var_post.loc.detach().numpy()[0], np.exp(self.var_post.scale_raw.detach().numpy()[0])
                # ))
                # print('Average dists: weight N({:.2f}, {:.2f}), bias N({:.2f}, {:.2f})'.format(
                #     sum([v[0] for v in vf_history])/log_period,
                #     np.exp(sum([v[1] for v in vf_history])/log_period),
                #     sum([v[2] for v in vf_history])/log_period,
                #     np.exp(sum([v[3] for v in vf_history])/log_period)
                # ))

                # log learning rate
                # message += ', LR: '+str(self.lr_scheduler.get_last_lr())

                # update the best VFs based on average loss
                if sum(loss_history)/log_period < min_criterion:
                    min_criterion = sum(loss_history)/log_period
                    self.best_vfs = copy.deepcopy(self.var_post)
                    self.logger.info('updated best variational factors.')


            # # go one iter back if non-psd
            # if self.unknown_err:
            #     self.var_post =copy.deepcopy(self.best_vfs)  # set back to best seen

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

        # # define posterior with average weights of final iters
        # self.av_var_post = copy.deepcopy(self.var_post)
        # self.av_var_post.loc = torch.tensor([
        #     sum([v[1] for v in vf_history])/log_period,
        #     sum([v[0] for v in vf_history])/log_period,
        # ])
        # self.av_var_post.scale_raw = torch.tensor([
        #     sum([v[3] for v in vf_history])/log_period,
        #     sum([v[2] for v in vf_history])/log_period,
        # ])



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
            elbo = log_posterior_num - log_var_post
        elbo = elbo.reshape(self.L)
        assert elbo.ndim == 1 and elbo.shape[0] == self.L
        return - torch.mean(elbo)

    # -------------------------
    # ------ setup model ------
    def _setup_model_inference(
        self, sys, lambda_, loss, prior_dict,
        optimizer, lr, lr_decay,
        layer_sizes, nonlinearity_hidden, nonlinearity_output,
        controller_type, n_xi, l, x_init, u_init,
        # VI properties
        num_vfs, vf_init_std, vf_cov_type, vf_param_dists, L
    ):
        self.num_vfs, self.L = num_vfs, L
        """define a generic Gibbs posterior"""
        # only used to know how many parameters we have
        self.generic_Gibbs = GibbsPosterior(
            controller_type=controller_type,
            sys=copy.deepcopy(sys), loss_fn=loss,
            lambda_=lambda_, prior_dict=prior_dict,
            initialization_std=0.1, # for initializing REN. not importane
            # NN
            layer_sizes=layer_sizes,
            nonlinearity_hidden=nonlinearity_hidden,
            nonlinearity_output=nonlinearity_output,
            # REN
            n_xi=n_xi, l=l, x_init=x_init, u_init=u_init,
            # Misc
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

    # -------------------------

    # def multi_rollout(self, data):
    #     """
    #     rollout using current VFs.
    #     Tracks grads.
    #     """
    #     data = to_tensor(data)
    #     # if len(data.shape)==2:
    #     #     data = torch.reshape(data, (data, *data.shape))

    #     res_xs, res_ys, res_us = [], [], []
    #     for particle_num in range(self.num_particles):
    #         particle = self.particles[particle_num, :]
    #         # set this particle as params of a controller
    #         posterior_copy = self.posterior
    #         cl_system = posterior_copy.get_forward_cl_system(particle)
    #         # rollout
    #         xs, ys, us = cl_system.multi_rollout(data)
    #         res_xs.append(xs)
    #         res_ys.append(ys)
    #         res_us.append(us)
    #     assert len(res_xs) == self.num_particles
    #     return res_xs, res_ys, res_us

    # def eval_rollouts(self, data, get_full_list=False, loss_fn=None):
    #     """
    #     evaluates several rollouts given by 'data'.
    #     if 'get_full_list' is True, returns a list of losses for each particle.
    #     o.w., returns average loss of all particles.
    #     if 'loss_fn' is None, uses the bounded loss function as in Gibbs posterior.
    #     loss_fn can be provided to evaluate the dataset using the original unbounded loss.
    #     """
    #     with torch.no_grad():
    #         losses=[None]*self.num_particles
    #         res_xs, _, res_us = self.multi_rollout(data)
    #         for particle_num in range(self.num_particles):
    #             if loss_fn is None:
    #                 losses[particle_num] = self.posterior.loss_fn.forward(
    #                     res_xs[particle_num], res_us[particle_num]
    #                 ).item()
    #             else:
    #                 losses[particle_num] = loss_fn.forward(
    #                     res_xs[particle_num], res_us[particle_num]
    #                 ).item()
    #     if get_full_list:
    #         return losses
    #     else:
    #         return sum(losses)/self.num_particles


    # def _vectorize_pred_dist(self, pred_dist):
    #     multiv_normal_batched = pred_dist.dists
    #     normal_batched = torch.distributions.Normal(multiv_normal_batched.mean, multiv_normal_batched.stddev)
    #     return EqualWeightedMixtureDist(normal_batched, batched=True, num_dists=multiv_normal_batched.batch_shape[0])


# ---------------------------------------
import math
from collections import OrderedDict
from pyro.distributions import Normal

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

        assert vf_cov_type in ['diag', 'full']

        self.param_idx_ranges = OrderedDict()

        idx_start = 0
        for name, shape in named_param_shapes.items():
            assert len(shape) == 1
            idx_end = idx_start + shape[0]
            self.param_idx_ranges[name] = (idx_start, idx_end)
            idx_start = idx_end

        param_shape = torch.Size((idx_start,))

        # initialize VI
        if vf_param_dists is None: # initialize randomly
            self.loc = torch.nn.Parameter(
                torch.normal(0.0, vf_init_std, size=param_shape, device=device))
            if vf_cov_type == 'diag':
                self.scale_raw = torch.nn.Parameter(
                    torch.normal(math.log(0.1), vf_init_std, size=param_shape, device=device))
                self.dist_fn = lambda: Normal(self.loc, self.scale_raw.exp()).to_event(1)
            if vf_cov_type == 'full':
                self.tril_cov = torch.nn.Parameter(
                    torch.diag(torch.ones(param_shape, device=device).uniform_(0.05, 0.1)))
                self.dist_fn = lambda: torch.distributions.MultivariateNormal(
                    loc=self.loc, scale_tril=torch.tril(self.tril_cov))
        else: # initialize from given vf_param_dists
            # go through all parameters by name and initialize
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
            if vf_cov_type == 'diag':
                self.scale_raw = torch.nn.Parameter(torch.tensor(scale_raw).float().to(device))
                self.dist_fn = lambda: Normal(self.loc, self.scale_raw.exp()).to_event(1)
            if vf_cov_type == 'full':
                raise NotImplementedError


    def forward(self):
        return self.dist_fn()

    def rsample(self, sample_shape=torch.Size()):
        return self.forward().rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()):
        return self.forward().sample(sample_shape)

    def log_prob(self, value):
        return self.forward().log_prob(value)

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