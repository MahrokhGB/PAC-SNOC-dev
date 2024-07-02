import torch
from assistive_functions import to_tensor

class LQLossFH():
    def __init__(self, Q, R, loss_bound, sat_bound, xbar=None):
        self.Q, self.R = Q, R
        self.Q = to_tensor(self.Q)
        self.R = to_tensor(self.R)
        assert len(self.Q.shape)==2 and self.Q.shape[0] == self.Q.shape[1]
        assert (not hasattr(self.R, "__len__")) or len(self.R.shape)==2  # int or square matrix
        self.loss_bound, self.sat_bound = loss_bound, sat_bound
        if not self.loss_bound is None:
            assert not self.sat_bound is None
            self.loss_bound = to_tensor(self.loss_bound)
        if not self.sat_bound is None:
            assert not self.loss_bound is None
            self.sat_bound = to_tensor(self.sat_bound)
        self.xbar = xbar
        if not self.xbar is None:
            self.xbar = to_tensor(self.xbar)
            self.xbar = self.xbar.reshape(self.Q.shape[0], 1)

    def forward(self, xs, us):
        '''
        compute loss
        Args:
            - xs: tensor of shape (S, T, num_states)
            - us: tensor of shape (S, T, num_inputs)
        '''
        if self.xbar is not None:
            xs = xs - self.xbar.repeat(xs.shape[0], 1, 1)
        # batch
        xs = xs.reshape(*xs.shape, 1)
        us = us.reshape(*us.shape, 1)
        # batched multiplication
        xTQx = torch.matmul(torch.matmul(xs.transpose(-1, -2), self.Q), xs)         # shape = (S, T, 1, 1)
        uTRu = torch.matmul(torch.matmul(us.transpose(-1, -2), self.R), us)         # shape = (S, T, 1, 1)
        # average over the time horizon
        loss_x = torch.sum(xTQx, 1) / xs.shape[1]    # shape = (S, 1, 1)
        loss_u = torch.sum(uTRu, 1) / us.shape[1]    # shape = (S, 1, 1)
        loss_val = loss_x + loss_u
        # bound
        if self.sat_bound is not None:
            loss_val = torch.tanh(loss_val/self.sat_bound)  # shape = (S, 1, 1)
        if self.loss_bound is not None:
            loss_val = self.loss_bound * loss_val           # shape = (S, 1, 1)
        # average over the samples
        loss_val = torch.sum(loss_val, 0)/xs.shape[0]       # shape = (1, 1)
        return loss_val
