import torch, sys
from numpy import random
from config import BASE_DIR
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
            theta=torch.zeros(sys.num_inputs, sys.num_states),
            bias=torch.zeros(sys.num_inputs, 1)
        )
    else:
        raise NotImplementedError

    return generic_controller

# ---------- CONTROLLER ----------
from assistive_functions import to_tensor
class AffineController:
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
