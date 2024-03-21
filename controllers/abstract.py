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

    def multi_rollout(self, data):
        (S, T, num_states) = data.shape
        assert num_states==self.sys.num_states

        for sample_num in range(S):
            if self.sys.__class__.__name__=='SystemRobots':
                x_tmp, y_tmp, u_tmp = self.sys.rollout(
                    controller=self.controller,
                    data=data[sample_num, :, :], train=True
                )
            else:
                x_tmp, y_tmp, u_tmp = self.sys.rollout(
                    controller=self.controller,
                    data=data[sample_num, :, :]
                )
            if sample_num==0:
                xs = x_tmp.reshape(1, *x_tmp.shape)
                ys = y_tmp.reshape(1, *y_tmp.shape) if not y_tmp is None else None
                us = u_tmp.reshape(1, *u_tmp.shape)
            else:
                xs = torch.cat((xs, x_tmp.reshape(1, *x_tmp.shape)), 0)
                ys = torch.cat((ys, y_tmp.reshape(1, *y_tmp.shape)), 0) if not y_tmp is None else None
                us = torch.cat((us, u_tmp.reshape(1, *u_tmp.shape)), 0)

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
import numpy as np
class affine_controller:
    def __init__(self, theta, bias=None):
        # theta.shape = (num_inputs, num_states), bias.shape=(num_inputs, 1)
        if not (bias is None or isinstance(bias, np.ndarray)):
            bias = np.array(bias)
        self.theta = theta
        self.bias = bias.reshape(theta.shape[0], 1) if bias is not None else np.zeros((theta.shape[0], 1))
        self.num_states = theta.shape[1]

    def forward(self, what):
        assert what.shape[0] == self.num_states, what.shape
        return np.matmul(self.theta, what)+self.bias

    def set_vector_as_params(self, vec):
        # last element is bias, the rest is theta
        assert len(vec) == len(self.theta)+len(self.bias)
        self.theta = vec[:-1].reshape(self.theta.shape)
        self.bias = vec[-1].reshape(self.bias.shape)
