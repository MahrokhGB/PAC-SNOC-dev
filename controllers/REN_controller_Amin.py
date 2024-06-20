#!/usr/bin/env python
import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from assistive_functions import WrapLogger, _check_data_dim
from config import device

'''

'''

# REN implementation in the acyclic version
# See paper: "Recurrent Equilibrium Networks: Flexible dynamic models with guaranteed stability and robustness"
# TODO: change init method
class PsiU(nn.Module):
    def __init__(
        self, dim_in, dim_out, dim_xi, l, initialization_std, xi_init = None
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_xi = dim_xi
        self.dim_out = dim_out
        self.l = l
        # initialize internal state
        self.xi = xi_init if not xi_init is None else torch.zeros(1, self.dim_xi)
        self.xi = _check_data_dim(self.xi, (1, self.dim_xi))
        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        self.X_shape = (2*self.dim_xi+self.l, 2*self.dim_xi+self.l)
        self.Y_shape = (self.dim_xi, self.dim_xi)
        # NN state dynamics:
        self.B2_shape = (self.dim_xi, self.dim_in)
        # NN output:
        self.C2_shape = (self.dim_out, self.dim_xi)
        self.D21_shape = (self.dim_out, self.l)
        self.D22_shape = (self.dim_out, self.dim_in)
        # v signal:
        self.D12_shape = (self.l, self.dim_in)

        # define training nn params
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
        for name in self.training_param_names:
            shape = getattr(self, name+'_shape')
            assert len(shape) == 2
            # define each param as nn.Parameter
            setattr(
                self, name+'_vec',
                nn.Parameter(
                    (torch.randn(shape[0] * shape[1], device=device)*initialization_std)
                )
            )


        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(dim_xi, dim_xi).to(device)
        self.B1 = torch.zeros(dim_xi, l).to(device)
        self.E = torch.zeros(dim_xi, dim_xi).to(device)
        self.Lambda = torch.ones(l).to(device)
        self.C1 = torch.zeros(l, dim_xi).to(device)
        self.D11 = torch.zeros(l, l).to(device)
        self.set_model_param()

    def set_model_param(self):
        # convert vectorized training params to matrices
        for name in self.training_param_names:
            shape = getattr(self, name+'_shape')
            vectorized_param = getattr(self, name+'_vec')
            setattr(self, name, vectorized_param.reshape(shape))
        # dependent params
        l = self.l
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2*self.dim_xi+l).to(device)
        h1, h2, h3 = torch.split(H, [self.dim_xi, l, self.dim_xi], dim=0)
        H11, H12, H13 = torch.split(h1, [self.dim_xi, l, self.dim_xi], dim=1)
        H21, H22, _ = torch.split(h2, [self.dim_xi, l, self.dim_xi], dim=1)
        H31, H32, H33 = torch.split(h3, [self.dim_xi, l, self.dim_xi], dim=1)
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

    def forward(self, u_in):
        '''
        returns:
        - y_out of shape (batch_size, 1, self.dim_out)

        sizes of variables (for debugging):
        - w: (batch_size, 1, self.dim_in)
        - xi: (batch_size, 1, self.dim_xi)
        - u_in: (batch_size, 1, self.dim_in)
        '''
        batch_size = u_in.shape[0]

        vec = torch.zeros(self.l).to(device)
        vec[0] = 1
        w = torch.zeros(batch_size, 1, self.l).to(device)
        v = F.linear(self.xi, self.C1[0,:]) + F.linear(u_in, self.D12[0,:])
        assert v.shape==(batch_size, 1)
        # update each row of w using Eq. (8) with a lower triangular D11
        w = w + (vec * torch.tanh(v/self.Lambda[0])).reshape(batch_size, 1, self.l)
        for i in range(1, self.l):
            vec = torch.zeros(self.l).to(device)  # mask -- all zeros except in one position
            vec[i] = 1
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + F.linear(u_in, self.D12[i,:])  # self.bv[i]
            w = w + (vec * torch.tanh(v/self.Lambda[i])).reshape(batch_size, 1, self.l)
        # compute next state using Eq. 18
        self.xi = F.linear(
            F.linear(self.xi, self.F) + F.linear(w, self.B1) + F.linear(u_in, self.B2),
            self.E.inverse()
        )
        y_out = F.linear(self.xi, self.C2) + F.linear(w, self.D21) + F.linear(u_in, self.D22)  # + self.bu
        return y_out

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
