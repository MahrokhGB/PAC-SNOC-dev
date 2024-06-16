import torch
from assistive_functions import to_tensor

# ---------- SYSTEM ----------
class LTI_system:
    def __init__(self, A, B, C, x_init):
        self.A, self.B, self.C = to_tensor(A), to_tensor(B), to_tensor(C)
        self.x_init = to_tensor(x_init)

        # Dimensions
        self.num_states = self.A.shape[0]
        self.num_inputs = self.B.shape[1]
        self.num_outputs = self.C.shape[0]
        # Check matrices
        assert self.A.shape == (self.num_states, self.num_states)
        assert self.B.shape == (self.num_states, self.num_inputs)
        assert self.C.shape == (self.num_outputs, self.num_states)
        assert self.x_init.shape == (self.num_states, 1)

    # # simulation
    def multi_rollout(self, controller, data):
        return self.rollout(controller, data)
    #     (S, T, num_states) = data.shape
    #     assert num_states
    #     ys = torch.zeros(S, T, self.num_outputs)
    #     xs = torch.zeros(S, T, self.num_states)
    #     us = torch.zeros(S, T, self.num_inputs)
    #     # simulate for all disturbance samples
    #     for sample_ind in range(S):
    #         states, resp, inputs = self.rollout(
    #             controller,
    #             data[sample_ind, :, :]
    #         )
    #         ys[sample_ind, :, :] = resp
    #         xs[sample_ind, :, :] = states
    #         us[sample_ind, :, :] = inputs
    #     return xs, ys, us

    def rollout(self, controller, data, **kwargs):
        """
        rollout with state feedback controller
        inputs:
        - data: batch of T disturbance samples, with shape (batch_size, T, num_states)
        """
        # check shape is (batch_size, T, num_states)
        data = to_tensor(data)
        if len(data.shape) == 1:
            data = torch.reshape(data, (1, -1, 1))
        elif len(data.shape) == 2:
            data = torch.reshape(data, (1, *data.shape))
        assert len(data.shape) == 3
        (batch_size, T, num_states) = data.shape
        assert num_states == self.num_states

        # Simulate
        xs = (data[:, 0:1, :] + self.x_init) #.reshape(batch_size, 1, num_states)
        us = controller.forward(xs[:, 0:1, :]) #.reshape(batch_size, 1, num_states)
        ys = torch.matmul(self.C, xs[:, 0:1, :]) #.reshape(batch_size, 1, self.num_outputs)
        for t in range(1, T):
            # state and response
            xs = torch.cat(
                (
                    xs,
                    torch.matmul(self.A, xs[:, t-1:t, :]) + torch.matmul(self.B, us[:, t-1:t, :]) + data[:, t:t+1, :]),
                1
            )
            ys = torch.cat(
                (ys, torch.matmul(self.C, xs[:, t:t+1, :])),
                1
            )
            us = torch.cat(
                (us, controller.forward(xs[:, t:t+1, :])),
                1
            )
        return xs, ys, us
