import torch, sys
from numpy import random
from config import BASE_DIR, device
sys.path.append(BASE_DIR)


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

    def parameter_shapes(self):
        return

    def named_parameters(self):
        return


# can be removed and use vectorized
from controllers.vectorized_controller import ControllerVectorized
class LinearController(ControllerVectorized):
    def __init__(self, num_states, num_inputs, requires_bias={'out':True, 'hidden':True}):
        super().__init__(
            num_states, num_inputs, layer_sizes=[], nonlinearity_hidden=None,
            nonlinearity_output=None, requires_bias=requires_bias
        )


# ---------- CONTROLLER ----------
# can be removed
from assistive_functions import to_tensor
class affine_controller:
    def __init__(self, theta, bias=None):
        # theta is a tensor of shape = (num_inputs, num_states)
        self.theta = to_tensor(theta)
        if len(self.theta.shape)==1:
            self.theta = self.theta.reshape(1, -1)
        self.num_inputs, self.num_states = self.theta.shape
        # bias is a tensor of shape=(num_inputs, 1)
        self.bias = torch.zeros((theta.shape[0], 1)) if bias is None else to_tensor(bias)
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
        return torch.matmul(self.theta, what)+self.bias

    def set_vector_as_params(self, vec):
        # last element is bias, the rest is theta
        assert len(vec) == len(self.theta)+len(self.bias)
        self.theta = vec[:-1].reshape(self.theta.shape)
        self.bias = vec[-1].reshape(self.bias.shape)
