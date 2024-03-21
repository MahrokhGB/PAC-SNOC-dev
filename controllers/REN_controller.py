#!/usr/bin/env python
import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from assistive_functions import WrapLogger
from config import device


# REN implementation in the acyclic version
# See paper: "Recurrent Equilibrium Networks: Flexible dynamic models with guaranteed stability and robustness"
class PsiU(nn.Module):
    def __init__(
        self, num_states, num_inputs, n_xi,
        l, train_method, initialization_std
    ):
        super().__init__()
        self.num_states = num_states
        self.n_xi = n_xi
        self.l = l
        self.num_inputs = num_inputs
        self.train_method = train_method
        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        self.X_shape = (2*n_xi+l, 2*n_xi+l)
        self.Y_shape = (n_xi, n_xi)
        # NN state dynamics:
        self.B2_shape = (n_xi, self.num_states)
        # NN output:
        self.C2_shape = (self.num_inputs, n_xi)
        self.D21_shape = (self.num_inputs, l)
        self.D22_shape = (self.num_inputs, self.num_states)
        # v signal:
        self.D12_shape = (l, self.num_states)

        # define training nn params
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
        for name in self.training_param_names:
            shape = getattr(self, name+'_shape')
            assert len(shape) == 2
            if self.train_method == 'empirical':
                # define each param as nn.Parameter
                setattr(
                    self, name+'_vec',
                    nn.Parameter(
                        (torch.randn(shape[0] * shape[1], device=device)*initialization_std)
                    )
                )
            elif self.train_method == 'SVGD':
                setattr(
                    self, name+'_vec',
                    torch.normal(
                        0, initialization_std, size=(shape[0] * shape[1],),
                        device=device, requires_grad=False, 
                        dtype=torch.float32
                    )
                )
            else:
                raise NotImplementedError

        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi).to(device)
        self.B1 = torch.zeros(n_xi, l).to(device)
        self.E = torch.zeros(n_xi, n_xi).to(device)
        self.Lambda = torch.ones(l).to(device)
        self.C1 = torch.zeros(l, n_xi).to(device)
        self.D11 = torch.zeros(l, l).to(device)
        self.set_model_param()

    def set_model_param(self):
        # convert vectorized training params to matrices
        for name in self.training_param_names:
            shape = getattr(self, name+'_shape')
            vectorized_param = getattr(self, name+'_vec')
            setattr(self, name, vectorized_param.reshape(shape))
        # dependent params
        n_xi = self.n_xi
        l = self.l
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2*n_xi+l).to(device)
        h1, h2, h3 = torch.split(H, [n_xi, l, n_xi], dim=0)
        H11, H12, H13 = torch.split(h1, [n_xi, l, n_xi], dim=1)
        H21, H22, _ = torch.split(h2, [n_xi, l, n_xi], dim=1)
        H31, H32, H33 = torch.split(h3, [n_xi, l, n_xi], dim=1)
        P = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + P + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, t, w, xi):
        vec = torch.zeros(self.l).to(device)
        vec[0] = 1
        epsilon = torch.zeros(self.l).to(device)
        v = F.linear(xi, self.C1[0,:]) + F.linear(w, self.D12[0,:])  # + self.bv[0]
        epsilon = epsilon + vec * torch.tanh(v/self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l).to(device)  # mask -- all zeros except in one position
            vec[i] = 1
            v = F.linear(xi, self.C1[i,:]) + F.linear(epsilon, self.D11[i,:]) + F.linear(w, self.D12[i,:])  # self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v/self.Lambda[i])
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon, self.B1) + F.linear(w, self.B2)  # + self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + F.linear(w, self.D22)  # + self.bu
        return u, xi_

    def parameter_shapes(self):
        param_dict = OrderedDict(
            (name+'_vec', getattr(self, name+'_vec').shape) for name in self.training_param_names
        )
        return param_dict

    def named_parameters(self):
        param_dict = OrderedDict(
            (name+'_vec', getattr(self, name+'_vec')) for name in self.training_param_names
        )
        return param_dict


class RENController(nn.Module):
    def __init__(
        self, noiseless_forward, num_states, num_inputs,
        n_xi, l, x_init, u_init, initialization_std,
        output_amplification,
        train_method='empirical', logger=None
    ):
        super().__init__()
        assert train_method in ['empirical', 'SVGD']
        self.num_states = num_states    # was self.n
        self.num_inputs = num_inputs  # was self.m
        self.psi_u = PsiU(
            self.num_states, self.num_inputs, n_xi, l, train_method,
            initialization_std=initialization_std
        )  # This is the REN
        self.noiseless_forward = noiseless_forward  # this is the model of the system
        # set initial conditions
        self.t = 0
        self.x_init, self.u_init = x_init, u_init
        self.last_y = copy.deepcopy(self.x_init)
        self.last_u = copy.deepcopy(self.u_init)
        self.last_xi = torch.zeros(self.psi_u.n_xi).to(device)
        self.logger = WrapLogger(logger)
        self.output_amplification = output_amplification

    def reset(self):
        self.t = 0
        self.last_y = copy.deepcopy(self.x_init)
        self.last_u = copy.deepcopy(self.u_init)
        self.last_xi = torch.zeros(self.psi_u.n_xi).to(device)

    def forward(self, y_):
        x_noiseless = self.noiseless_forward(self.t, self.last_y, self.last_u)
        w_ = y_ - x_noiseless  # reconstruction of the noise
        u_, xi_ = self.psi_u.forward(self.t, w_, self.last_xi)
        u_ = u_*self.output_amplification
        self.last_y, self.last_u = y_, u_
        self.last_xi = xi_
        self.t += 1
        return u_

    # functions for handling parameters access
    def parameter_shapes(self):
        return self.psi_u.parameter_shapes()

    def named_parameters(self):
        return self.psi_u.named_parameters()

    def parameters(self):
        return list(self.named_parameters().values())

    def set_parameter(self, name, value):
        current_val = getattr(self.psi_u, name)
        if self.psi_u.train_method == 'SVGD':
            value = value.reshape(current_val.shape)
        elif self.psi_u.train_method == 'empirical':
            value = torch.nn.Parameter(value.reshape(current_val.shape))
        else:
            raise NotImplementedError
        # if value.is_leaf:
        #     value.requires_grad=current_val.requires_grad

        setattr(self.psi_u, name, value)
        self.psi_u.set_model_param()    # update dependent params

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def parameters_as_vector(self):
        return torch.cat(self.parameters(), dim=-1)

    def set_parameters_as_vector(self, value):
        # value is reshaped to the parameter shape
        idx = 0
        for name, shape in self.parameter_shapes().items():
            idx_next = idx + shape[-1]
            # select indx
            if value.ndim == 1:
                value_tmp = value[idx:idx_next]
            elif value.ndim == 2:
                value_tmp = value[:, idx:idx_next]
            else:
                raise AssertionError
            # set
            if self.psi_u.train_method=='SVGD':
                self.set_parameter(name, value_tmp)
            elif self.psi_u.train_method=='empirical':
                with torch.no_grad():
                    self.set_parameter(name, value_tmp)
            else:
                raise NotImplementedError
            idx = idx_next
        assert idx_next == value.shape[-1]

    def print_params(self):
        self.logger.info(self.named_parameters())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
