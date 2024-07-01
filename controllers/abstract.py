import torch, sys
from numpy import random
from config import BASE_DIR
sys.path.append(BASE_DIR)
from config import device


# The `CLSystem` class is a neural network module that performs multi-rollout simulations using a
# given system and controller.
class CLSystem(torch.nn.Module):
    def __init__(self, sys, controller, random_seed):
        super().__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            self.random_state = random.RandomState(random_seed)
        else:
            self.random_state = random.RandomState(0)
        self.sys=sys
        self.controller=controller

    def rollout(self, data):
        assert len(data.shape)==3
        (S, T, num_states) = data.shape
        assert num_states==self.sys.num_states

        if self.sys.__class__.__name__=='SystemRobots':
            xs, ys, us= self.sys.rollout(
                controller=self.controller,
                data=data, train=True
            )
        else:
            xs, ys, us = self.sys.rollout(
                controller=self.controller,
                data=data
            )
        assert xs.shape==(S, T, num_states), xs.shape
        return xs, ys, us



from controllers.REN_controller import RENController
def get_controller(
    controller_type, sys,
    # REN controller
    n_xi=None, l=None, x_init=None, u_init=None, initialization_std=None,
):
    if controller_type == 'REN':
        assert not (n_xi is None or l is None or x_init is None or u_init is None)
        generic_controller = RENController(
            noiseless_forward=sys.noiseless_forward,
            output_amplification=20,
            num_states=sys.num_states, num_inputs=sys.num_inputs,
            n_xi=n_xi, l=l, x_init=x_init, u_init=u_init,
            train_method='SVGD', initialization_std=initialization_std
        )
    elif controller_type=='Affine':
        generic_controller = AffineController(
            weight=torch.zeros(sys.num_inputs, sys.num_states, device=device, dtype=torch.float32),
            bias=torch.zeros(sys.num_inputs, 1, device=device, dtype=torch.float32)
        )
    else:
        raise NotImplementedError

    return generic_controller

# ---------- CONTROLLER ----------
from collections import OrderedDict
from assistive_functions import to_tensor
class AffineController:
    def __init__(self, weight, bias=None):
        # weight is a tensor of shape = (num_inputs, num_states)
        self.weight = to_tensor(weight)
        if len(self.weight.shape)==1:
            self.weight = self.weight.reshape(1, -1)
        self.num_inputs, self.num_states = self.weight.shape
        # bias is a tensor of shape=(num_inputs, 1)
        self.bias = torch.zeros((weight.shape[0], 1)) if bias is None else to_tensor(bias)
        if len(self.bias.shape)==1:
            self.bias = self.bias.reshape(-1, 1)
        assert self.bias.shape==(self.num_inputs, 1)


    def forward(self, what):
        # what must be of shape (batch_size, num_states, 1)
        what = to_tensor(what)
        if len(what.shape)==1:
            what = what.reshape(1, -1, 1)
        if len(what.shape)==2:
            what = what.reshape(1, *what.shape)
        assert what.shape[1:]==torch.Size([self.num_states, self.num_inputs]), what.shape
        return torch.matmul(self.weight, what)+self.bias

    def set_parameters_as_vector(self, vec):
        # last element is bias, the rest is weight
        vec = vec.flatten()
        assert len(vec) == len(self.weight)+len(self.bias)
        self.weight = vec[:len(self.weight)].reshape(self.weight.shape)
        self.bias = vec[len(self.weight):].reshape(self.bias.shape)

    def parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in ['weight', 'bias']
        )
        return param_dict

    def named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in ['weight', 'bias']
        )
        return param_dict
