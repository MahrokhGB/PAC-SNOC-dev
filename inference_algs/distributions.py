import torch, sys, os, copy
from collections import OrderedDict
from torch.func import stack_module_state, functional_call
from pyro.distributions import Normal, Uniform
from torch.distributions import Distribution

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, BASE_DIR)

from config import device
from controllers.abstract import CLSystem, AffineController
from assistive_functions import to_tensor, WrapLogger
from controllers.REN_controller import RENController

class GibbsPosterior():

    def __init__(
        self, loss_fn, lambda_, prior_dict,
        # attributes of the CL system
        controller, sys,
        # misc
        logger=None, num_ensemble_models=1
    ):
        # set attributes
        self.lambda_ = to_tensor(lambda_)
        self.loss_fn = loss_fn
        self.logger = WrapLogger(logger)
        self.num_ensemble_models = num_ensemble_models

        # Controller params will be set during training and the resulting CL system is use for evaluation.
        self.generic_cl_system = CLSystem(sys, controller, random_seed=None)

        # init ensemble models
        self.ensemble_models = [copy.deepcopy(self.generic_cl_system)]*self.num_ensemble_models
        self.ensemble_params, self.ensemble_buffers = stack_module_state(self.ensemble_models)
        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        self.ensemble_base_model = copy.deepcopy(self.ensemble_models[0])
        self.ensemble_base_model = self.ensemble_base_model.to('meta')

        self._params = OrderedDict()
        self._param_dists = OrderedDict()
        # ------- set prior -------
        # set prior for REN controller
        if isinstance(self.generic_cl_system.controller, RENController):
            for name, shape in self.generic_cl_system.controller.parameter_shapes().items():
                # Gaussian prior
                if prior_dict['type'] == 'Gaussian':
                    if not (name+'_loc' in prior_dict.keys() or name+'_scale' in prior_dict.keys()):
                        self.logger.info('[WARNING]: prior for ' + name + ' was not provided. Replaced by default.')
                    dist = Normal(
                        loc=prior_dict.get(name+'_loc', 0)*torch.ones(shape, device=device),
                        scale=prior_dict.get(name+'_scale', 1)*torch.ones(shape, device=device)
                    )
                # Uniform prior
                elif prior_dict['type'] == 'Uniform':
                    raise NotImplementedError
                else:
                    raise NotImplementedError

                # set dist
                self._param_dist(name, dist.to_event(1))
        elif isinstance(self.generic_cl_system.controller, AffineController):
            for name, shape in self.generic_cl_system.controller.parameter_shapes().items():
                # Gaussian prior
                if prior_dict['type_'+name[0]].startswith('Gaussian'):
                    if not (name+'_loc' in prior_dict.keys() or name+'_scale' in prior_dict.keys()):
                        self.logger.info('[WARNING]: prior for ' + name + ' was not provided. Replaced by default.')
                    dist = Normal(
                        loc=prior_dict.get(name+'_loc', 0)*torch.ones(shape).flatten().to(device),
                        scale=prior_dict.get(name+'_scale', 1)*torch.ones(shape).flatten().to(device)
                    )
                # Uniform prior
                elif prior_dict['type_'+name[0]] == 'Uniform':
                    assert (name+'_low' in prior_dict.keys()) and (name+'_high' in prior_dict.keys())
                    dist = Uniform(
                        low=prior_dict[name+'_low']*torch.ones(shape).flatten().to(device),
                        high=prior_dict[name+'_high']*torch.ones(shape).flatten().to(device)
                    )
                else:
                    raise NotImplementedError
                # set dist
                print('Prior over '+ name + ':', dist)
                self._param_dist(name, dist.to_event(1))
        else:
            raise NotImplementedError

        # check that parameters in prior and controller are aligned
        for param_name_cont, param_name_prior in zip(self.generic_cl_system.controller.named_parameters().keys(), self._param_dists.keys()):
            assert param_name_cont == param_name_prior, param_name_cont + 'in controller did not match ' + param_name_prior + ' in prior'

        self.prior = CatDist(self._param_dists.values())

    def _log_prob_likelihood(self, params, train_data):
        # assert len(params.shape)<3
        # if len(params.shape)==1:
        #     params = params.reshape(1, -1)
        L = params.shape[0]

        for l_tmp in range(L):
            # set params to controller
            cl_system = self.generic_cl_system
            cl_system.controller.set_parameters_as_vector(
                params[l_tmp, :].reshape(1,-1)
            )
            # rollout
            xs, _, us = cl_system.rollout(train_data)
            # compute loss
            loss_val_tmp = self.loss_fn.forward(xs, us)
            if l_tmp==0:
                loss_val = [loss_val_tmp]
            else:
                loss_val.append(loss_val_tmp)
        loss_val = torch.cat(loss_val)
        assert loss_val.shape[0]==L and loss_val.shape[1]==1
        return loss_val

    def log_prob(self, params, train_data):
        '''
        params is of shape (L, -1)
        '''
        assert len(params.shape)<3
        if len(params.shape)==1:
            params = params.reshape(1, -1)
        L = params.shape[0]
        lpl = self._log_prob_likelihood(params, train_data)
        lpl = lpl.reshape(L)
        lpp = self._log_prob_prior(params)
        lpp = lpp.reshape(L)
        assert not (lpl.grad_fn is None or lpp.grad_fn is None)
        '''
        # NOTE: Must return lpp -self.lambda_ * lpl.
        # To have small prior effect, lamba must be large.
        # This makes the loss too large => divided by lambda^2.
        NOTE: To debug, remove the effect of the prior by returning -lpl
        '''
        # return - lpl
        # return 1/self.lambda_ * lpp - lpl
        return lpp - self.lambda_ * lpl

    def sample_params_from_prior(self, shape):
        # shape is torch.Size()
        return self.prior.sample(shape)

    def _log_prob_prior(self, params):
        return self.prior.log_prob(params)

    def _param_dist(self, name, dist):
        assert type(name) == str
        assert isinstance(dist, torch.distributions.Distribution)
        if isinstance(dist.base_dist, Normal):
            dist.base_dist.loc = dist.base_dist.loc.to(device)
            dist.base_dist.scale = dist.base_dist.scale.to(device)
        elif isinstance(dist.base_dist, Uniform):
            dist.base_dist.low = dist.base_dist.low.to(device)
            dist.base_dist.high = dist.base_dist.high.to(device)
        if name in list(self._param_dists.keys()):
            self.logger.info('[WARNING] name ' + name + 'was already in param dists')
        # assert name not in list(self._param_dists.keys())
        assert hasattr(dist, 'rsample')
        self._param_dists[name] = dist

        return dist

    def parameter_shapes(self):
        param_shapes_dict = OrderedDict()
        for name, dist in self._param_dists.items():
            param_shapes_dict[name] = dist.event_shape
        return param_shapes_dict

    def get_forward_cl_system(self, params):
        cl_system = self.generic_cl_system
        cl_system.controller.set_parameters_as_vector(params)
        return cl_system


# -------------------------
# -------------------------
from normflows.distributions import Target
# from torch.utils.data import DataLoader
class GibbsWrapperNF(Target):
    """
    Wrap given Gibbs distribution to be used in normflows
    """

    def __init__(
        self, target_dist, train_data, data_batch_size=None,
        prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)
    ):
        """Constructor

        Args:
          target_dist: Distribution to be approximated
          train_data: training data used to compute the Gibbs dist
          data_batch_size: size of data subsample to use in computing log prob
          prop_scale: Scale for the uniform proposal
          prop_shift: Shift for the uniform proposal
        """
        super().__init__(prop_scale=prop_scale, prop_shift=prop_shift)
        self.target_dist = target_dist
        self.train_data, self.data_batch_size = train_data, data_batch_size
        self.max_log_prob = 0.0
        if not self.data_batch_size is None:
            raise NotImplementedError   # TODO: random seed must be fixed across REN controller, ...

    def log_prob(self, z):
        """
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        # sample data batch
        # if self.data_batch_size < self.train_data.shape[0]:
        #     inds =
        return self.target_dist.log_prob(params=z, train_data=self.train_data)




# -------------------------
# -------------------------

class CatDist(Distribution):

    def __init__(self, dists, reduce_event_dim=True):
        assert all([len(dist.event_shape) == 1 for dist in dists])
        assert all([len(dist.batch_shape) == 0 for dist in dists])
        self.reduce_event_dim = reduce_event_dim
        self.dists = dists
        self._event_shape = torch.Size((sum([dist.event_shape[0] for dist in self.dists]),))

    def sample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, sample_fn='sample')

    def rsample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, sample_fn='rsample')

    def log_prob(self, value):
        idx = 0
        log_probs = []
        for dist in self.dists:
            n = dist.event_shape[0]
            if value.ndim == 1:
                val = value[idx:idx+n]
            elif value.ndim == 2:
                val = value[:, idx:idx + n]
            elif value.ndim == 2:
                val = value[:, :, idx:idx + n]
            else:
                raise NotImplementedError('Can only handle values up to 3 dimensions')
            log_probs.append(dist.log_prob(val))
            idx += n

        for i in range(len(log_probs)):
            if log_probs[i].ndim == 0:
                log_probs[i] = log_probs[i].reshape((1,))

        if self.reduce_event_dim:
            return torch.sum(torch.stack(log_probs, dim=0), dim=0)
        return torch.stack(log_probs, dim=0)

    def _sample(self, sample_shape, sample_fn='sample'):
        return torch.cat([getattr(d, sample_fn)(sample_shape).to(device) for d in self.dists], dim=-1)


class BlockwiseDist(Distribution):
    def __init__(self, priors):
        assert isinstance(priors, list)
        for prior in priors:
            assert isinstance(prior, Distribution)
        self.priors = priors
        self.num_priors = len(priors)

    def sample(self):
        res = torch.zeros(self.num_priors)
        for prior_num in range(self.num_priors):
            res[prior_num] = self.priors[prior_num].sample()
        return res
