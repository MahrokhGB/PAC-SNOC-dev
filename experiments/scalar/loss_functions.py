import torch
from assistive_functions import to_tensor, WrapLogger


class LQLossFH():
    def __init__(self, Q, R, T, loss_bound, sat_bound, logger=None):
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
        self.logger = WrapLogger(logger)

    def forward(self, xs, us):
        if not isinstance(xs, torch.Tensor):
            xs = to_tensor(xs)
        if not isinstance(us, torch.Tensor):
            us = to_tensor(us)
        (S, T, num_states) = xs.shape
        assert self.T == T and self.num_states == num_states
        (S, T, num_inputs) = us.shape
        assert self.T == T and self.num_inputs == num_inputs and xs.shape[0] == us.shape[0]

        for sample_num in range(S):
            for time in range(T):
                x = xs[sample_num, time, :].reshape(-1, 1)
                u = us[sample_num, time, :].reshape(-1, 1)
                xT = torch.transpose(x, 0, 1)
                uT = torch.transpose(u, 0, 1)
                xTQx = torch.matmul(torch.matmul(xT, self.Q), x)
                uTRu = torch.matmul(torch.matmul(uT, self.R), u)
                if xTQx[0,0] > 1e6:
                    self.logger.info(
                        '[WARN] xTQx too large at time {:3.0f} sample {:2.0f}'.format(time, sample_num)
                    )
                if uTRu[0,0] > 1e6:
                    self.logger.info(
                        '[WARN] uTRu too large at time {:3.0f} sample {:2.0f}'.format(time, sample_num)
                    )
                if time == 0:
                    loss_val = xTQx[0,0] + uTRu[0,0]
                else:
                    loss_val = loss_val + xTQx[0,0] + uTRu[0,0]
            # divide by horizon
            loss_val = loss_val/T
            # bound
            if self.sat_bound is not None:
                loss_val = torch.tanh(loss_val/self.sat_bound)
            if self.loss_bound is not None:
                loss_val = self.loss_bound * loss_val
            if sample_num == 0:
                loss_val_tot = loss_val
            else:
                loss_val_tot += loss_val
        loss_val_tot = loss_val_tot/S
        return loss_val_tot
