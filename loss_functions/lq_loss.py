import torch
from assistive_functions import to_tensor

class LQLossFH():
    def __init__(self, Q, R, T, loss_bound, sat_bound):
        if not isinstance(Q, torch.Tensor):
            Q = to_tensor(Q)
        if not isinstance(R, torch.Tensor):
            R = to_tensor(R)
        self.num_states = Q.shape[0]
        assert Q.shape == (self.num_states, self.num_states)
        self.num_inputs = R.shape[0]
        assert R.shape == (self.num_inputs, self.num_inputs)
        self.Q, self.R, self.T = Q, R, T
        self.loss_bound, self.sat_bound = loss_bound, sat_bound
        if not self.loss_bound is None:
            assert not self.sat_bound is None
            self.loss_bound = to_tensor(self.loss_bound)
        if not self.sat_bound is None:
            assert not self.loss_bound is None
            self.sat_bound = to_tensor(self.sat_bound)

    def forward(self, xs, us, xbar=None):
        '''
        compute loss
        Args:
            - xs: tensor of shape (S, T, num_states)
            - us: tensor of shape (S, T, num_inputs)
        '''
        if xbar is not None:
            xs = xs - xbar.repeat(xs.shape[0], 1, 1)
        # batch
        xs = xs.reshape(-1, self.T, self.num_states, 1)
        us = us.reshape(-1, self.T, self.num_inputs, 1)
        # batched multiplication
        xTQx = torch.matmul(torch.matmul(xs.transpose(-1, -2), self.Q), xs)         # shape = (S, T, 1, 1)
        uTRu = torch.matmul(torch.matmul(us.transpose(-1, -2), self.R), us)         # shape = (S, T, 1, 1)
        # average over the time horizon
        loss_x = torch.sum(xTQx, 1) / self.T    # shape = (S, 1, 1)
        loss_u = torch.sum(uTRu, 1) / self.T    # shape = (S, 1, 1)
        loss_val = loss_x + loss_u
        # bound
        if self.sat_bound is not None:
            loss_val = torch.tanh(loss_val/self.sat_bound)  # shape = (S, 1, 1)
        if self.loss_bound is not None:
            loss_val = self.loss_bound * loss_val           # shape = (S, 1, 1)
        # verage over the samples
        loss_val = torch.sum(loss_val, 0)/xs.shape[0]       # shape = (1, 1)
        return loss_val
