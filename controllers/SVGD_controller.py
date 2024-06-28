import torch, time, copy
from numpy import random
from config import device
from controllers.abstract import get_controller
from assistive_functions import WrapLogger, DummyLRScheduler, to_tensor
from inference_algs.svgd import SVGD, RBF_Kernel, IMQSteinKernel
from inference_algs.distributions import GibbsPosterior


class SVGDCont():
    def __init__(
        self, sys, train_d, lr, loss, prior_dict, controller_type,
        num_particles, initialization_std,
        initial_particles=None, kernel='RBF', bandwidth=None,
        random_seed=None, optimizer='Adam', batch_size=-1, lambda_=None,
        num_iter_fit=None, lr_decay=1.0, logger=None,
        # NN controller properties
        layer_sizes=None, nonlinearity_hidden=None, nonlinearity_output=None,
        # REN controller properties
        n_xi=None, l=None, x_init=None, u_init=None,
    ):
        """
        train_d: (ndarray) train disturbances - shape: (S, T, sys.num_states)
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
        self.num_particles = num_particles
        self.best_particles = None
        if batch_size < 1:
            self.batch_size = self.train_d.shape[0]
        else:
            self.batch_size = min(batch_size, self.train_d.shape[0])

        (S, T, num_states) = self.train_d.shape
        assert num_states == sys.num_states
        assert T == loss.T
        assert optimizer in ['Adam', 'SGD']
        assert controller_type in ['NN', 'REN']

        self.fitted, self.over_fitted = False, False
        self.unknown_err = False
        self.logger = WrapLogger(logger)

        """ --- Setup model & inference --- """
        self._setup_model_inference(
            sys=sys, lambda_=lambda_, loss=loss, prior_dict=prior_dict,
            initial_particles=initial_particles, kernel=kernel,
            bandwidth=bandwidth, optimizer=optimizer, lr=lr,
            lr_decay=lr_decay, layer_sizes=layer_sizes,
            nonlinearity_hidden=nonlinearity_hidden,
            nonlinearity_output=nonlinearity_output,
            controller_type=controller_type,
            n_xi=n_xi, l=l, x_init=x_init, u_init=u_init,
            initialization_std=initialization_std
        )

    def fit(self, over_fit_margin=None, cont_fit_margin=None, max_iter_fit=None,
            early_stopping=True, valid_data=None, log_period=500):
        """
        fits the hyper-posterior particles with SVGD

        Args:
            over_fit_margin: abrupt training if slope of valid RMSE over one log_period > over_fit_margin (set to None
             to disable early stopping)
            cont_fit_margin: continue training for more iters if slope of valid RMSE over one log_period<-cont_fit_margin
            max_iter_fit: max iters to extend training
            early_stopping: return model at an evaluated iteration with the lowest valid RMSE
            valid_data: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            log_period (int) number of steps after which to print stats
        """

        if early_stopping:
            self.best_particles = None
            min_criterion = 1e6

        # initial evaluation on train data
        message = 'Iter %d/%d' % (0, self.num_iter_fit)
        train_results = [self.eval_rollouts(self.train_d)]
        message += ' - Train Loss: {:2.4f}'.format(train_results[0])
        if valid_data is not None:
            # initial evaluation on validation data
            valid_results = [self.eval_rollouts(valid_data)]
            message += ', Valid Loss: {:2.4f}'.format(valid_results[0])
        self.logger.info(message)

        last_params = self.particles.detach().clone()  # params in the last iteration

        t = time.time()
        itr = 1
        while itr <= self.num_iter_fit:
            sel_inds = self.random_state.choice(self.train_d.shape[0], size=self.batch_size)
            task_dict_batch = self.train_d[sel_inds, :, :]
            # --- take a step ---
            try:
                self.svgd.step(self.particles, task_dict_batch)
            except Exception as e:
                self.logger.info('[Unhandled ERR] in SVGD step: ' + type(e).__name__ + '\n')
                self.logger.info(e)
                self.unknown_err = True

            # --- print stats ---
            if (itr % log_period == 0) and (not self.unknown_err):
                duration = time.time() - t
                t = time.time()
                message = 'Iter %d/%d - Time %.2f sec - SVGD Loss %.4f' % (
                    itr, self.num_iter_fit, duration, self.svgd.log_prob_particles
                )

                # evaluate on train set
                try:
                    train_res = self.eval_rollouts(self.train_d)
                    train_results.append(train_res)
                    message += ' - Train Loss: {:2.4f}'.format(train_res)

                except Exception as e:
                    self.logger.info('[Unhandled ERR] in eval train rollouts:' + type(e).__name__ + '\n')
                    self.logger.info(e)
                    self.unknown_err = True

                # if validation data is provided  -> compute the valid log-likelihood
                if valid_data is not None:
                    # evaluate on validation set
                    try:
                        valid_res = self.eval_rollouts(valid_data)
                        valid_results.append(valid_res)
                        message +=  ', Valid Loss: {:2.4f}'.format(valid_res)
                    except Exception as e:
                        message += '[Unhandled ERR] in eval valid rollouts:'
                        self.logger.info(e)
                        self.unknown_err = True

                    # check over-fitting
                    if not over_fit_margin is None:
                        if valid_results[-1]-valid_results[-2] >= over_fit_margin * log_period:
                            self.over_fitted = True
                            message += '\n[WARNING] model over-fitted'

                    # check continue training
                    if (not ((cont_fit_margin is None) or (max_iter_fit is None))) and (itr+log_period <= max_iter_fit) and (itr+log_period > self.num_iter_fit):
                        if valid_results[-1]-valid_results[-2] <= - abs(cont_fit_margin) * log_period:
                            self.num_iter_fit += log_period
                            message += '\n[Info] extended training'
                # log learning rate
                message += ', LR: '+str(self.lr_scheduler.get_last_lr())

                # update the best particles if early_stopping
                if early_stopping and itr > 1:
                    temp_res = train_results if valid_data is None else valid_results
                    if temp_res[-1] < min_criterion:
                        # self.logger.info('update best particle according to '+'train' if valid_data is None else 'valid')
                        min_criterion = temp_res[-1]
                        self.best_particles = self.particles.detach().clone()

                # log info
                self.logger.info(message)
                # update last params
                last_params = self.particles.detach().clone()

            # go one iter back if non-psd
            if self.unknown_err:
                self.particles = last_params.detach().clone()  # set back params to the previous iteration

            # stop training
            if self.over_fitted or self.unknown_err:
                break

            # update learning rate
            self.lr_scheduler.step()
            # go to next iter
            last_params = self.particles.detach().clone()  # num_rows = num_particles, columns are prior params. for variance, must apply softplus
            itr = itr+1

        self.fitted = True if not self.unknown_err else False

        # set back to the best particles if early stopping
        if early_stopping and (not self.best_particles is None):
            self.particles = self.best_particles

    def _setup_model_inference(
        self, sys, lambda_, loss, prior_dict, initial_particles,
        kernel, bandwidth, optimizer, lr, lr_decay, controller_type,
        layer_sizes, nonlinearity_hidden, nonlinearity_output,
        n_xi, l, x_init, u_init, initialization_std
    ):

        """define a generic controller"""
        # define a generic controller
        generic_controller = get_controller(
            controller_type=controller_type,
            initialization_std=initialization_std,
            # NN
            layer_sizes=layer_sizes,
            nonlinearity_hidden=nonlinearity_hidden,
            nonlinearity_output=nonlinearity_output,
            # REN
            n_xi=n_xi, l=l, x_init=x_init, u_init=u_init,
        )

        """define a generic Gibbs posterior"""
        self.posterior = GibbsPosterior(
            controller=generic_controller,
            sys=copy.deepcopy(sys), loss_fn=loss,
            lambda_=lambda_, prior_dict=prior_dict,
            logger=self.logger
        )

        """initialize particles"""
        if not initial_particles is None:
            # set given initial particles
            assert initial_particles.shape[0]==self.num_particles
            initial_sampled_particles = initial_particles.float().to(device)
            self.logger.info('[INFO] initialized particles with given value.')
        else:
            # sample initial particle locations from prior
            initial_sampled_particles = self.posterior.sample_params_from_prior(
                shape=(self.num_particles,)
            )
            self.logger.info('[INFO] initialized particles by sampling from the prior.')
        self.initial_particles = initial_sampled_particles
        # set particles to the initial value
        self.particles = self.initial_particles.detach().clone()
        self.particles.requires_grad = True

        self._setup_optimizer(optimizer, lr, lr_decay)

        """ Setup SVGD inference"""
        if kernel == 'RBF':
            kernel = RBF_Kernel(bandwidth=bandwidth)
        elif kernel == 'IMQ':
            kernel = IMQSteinKernel(bandwidth=bandwidth)
        else:
            raise NotImplementedError
        self.svgd = SVGD(self.posterior, kernel, optimizer=self.optimizer)

    def _setup_optimizer(self, optimizer, lr, lr_decay):
        assert hasattr(self, 'particles'), "SVGD must be initialized before setting up optimizer"
        assert self.particles.requires_grad

        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam([self.particles], lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD([self.particles], lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

    def rollout(self, data):
        """
        rollout using current particles.
        Tracks grads.
        """
        data = to_tensor(data)
        # if len(data.shape)==2:
        #     data = torch.reshape(data, (data, *data.shape))

        res_xs, res_ys, res_us = [], [], []
        for particle_num in range(self.num_particles):
            particle = self.particles[particle_num, :]
            # set this particle as params of a controller
            posterior_copy = self.posterior
            cl_system = posterior_copy.get_forward_cl_system(particle)
            # rollout
            xs, ys, us = cl_system.rollout(data)
            res_xs.append(xs)
            res_ys.append(ys)
            res_us.append(us)
        assert len(res_xs) == self.num_particles
        return res_xs, res_ys, res_us

    def eval_rollouts(self, data, get_full_list=False, loss_fn=None):
        """
        evaluates several rollouts given by 'data'.
        if 'get_full_list' is True, returns a list of losses for each particle.
        o.w., returns average loss of all particles.
        if 'loss_fn' is None, uses the bounded loss function as in Gibbs posterior.
        loss_fn can be provided to evaluate the dataset using the original unbounded loss.
        """
        with torch.no_grad():
            losses=[None]*self.num_particles
            res_xs, _, res_us = self.rollout(data)
            for particle_num in range(self.num_particles):
                if loss_fn is None:
                    losses[particle_num] = self.posterior.loss_fn.forward(
                        res_xs[particle_num], res_us[particle_num]
                    ).item()
                else:
                    losses[particle_num] = loss_fn.forward(
                        res_xs[particle_num], res_us[particle_num]
                    ).item()
        if get_full_list:
            return losses
        else:
            return sum(losses)/self.num_particles
