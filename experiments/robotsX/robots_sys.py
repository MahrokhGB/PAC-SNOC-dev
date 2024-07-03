import torch, copy
import torch.nn.functional as F

from config import device
from assistive_functions import to_tensor, check_data_dim


class SystemRobots(torch.nn.Module):
    def __init__(self, xbar, is_linear, x_init=None, u_init=None, k=1.0):
        """
        x_bar: initial point for all agents
        is_linear: if True, a linearized model of the system is used.
                   O.w., the model is non-linear. the non-linearity raises
                   from the dependence of friction on the speed.
        """
        super().__init__()
        self.is_linear = is_linear
        self.n_agents = int(xbar.shape[0]/4)
        self.num_states = 4*self.n_agents
        self.num_inputs = 2*self.n_agents
        self.h = 0.05
        self.mass = 1.0
        self.k = k
        self.b = 1.0
        self.b2 = None if self.is_linear else 0.1
        m = self.mass
        self.B = torch.kron(torch.eye(self.n_agents),
                            torch.tensor([[0, 0],
                                          [0., 0],
                                          [1/m, 0],
                                          [0, 1/m]])
                            ) * self.h
        self.B = self.B.to(device)
        self.xbar = xbar
        self.x_init = copy.deepcopy(xbar) if x_init is None else x_init
        self.u_init = torch.zeros(self.num_inputs).to(device) if u_init is None else u_init
        self.x_init = self.x_init.reshape(1, -1)
        self.u_init = self.u_init.reshape(1, -1)

        self._A1 = torch.eye(4*self.n_agents).to(device)
        self._A2 = torch.cat((torch.cat((torch.zeros(2,2),
                                   torch.eye(2)
                                   ), dim=1),
                        torch.cat((torch.diag(torch.tensor([-self.k/self.mass, -self.k/self.mass])),
                                   torch.diag(torch.tensor([-self.b/self.mass, -self.b/self.mass]))
                                   ),dim=1),
                        ),dim=0)
        self._A2 = torch.kron(torch.eye(self.n_agents), self._A2).to(device)
        self.A = self._A1 + self.h * self._A2

    def A_nonlin(self, x):
        assert not self.is_linear
        mask = torch.tensor([[0, 0], [1, 1]]).repeat(self.n_agents, 1).to(device)
        A3 = torch.norm(
            x.view(-1, 2 * self.n_agents, 2) * mask, dim=-1, keepdim=True
        )               # shape = (batch_size, 4, 1)
        A3 = torch.kron(
            A3, torch.ones(2, 1).to(device)
        )               # shape = (batch_size, 8, 1)
        A3 = -self.b2 / self.mass * torch.diag_embed(
            A3.squeeze(dim=-1), offset=0, dim1=-2, dim2=-1
        ).to(device)    # shape = (batch_size, 8, 8)
        A = self._A1 + self.h * (self._A2 + A3)
        return A

    def noiseless_forward(self, t, x, u):
        # check sizes
        x = check_data_dim(x, vec_dim=(1, self.num_states))
        u = check_data_dim(u, vec_dim=(1, self.num_inputs))
        batch_size = x.shape[0]
        assert batch_size==u.shape[0], 'batch sizes of x and u are different.'

        if self.is_linear:
            # x is batched but A is not => can use F.linear to compute xA^T
            f = F.linear(x - self.xbar, self.A) + F.linear(u, self.B) + self.xbar
        else:
            # A depends on x, hence is batched. perform batched matrix multiplication
            f = torch.bmm(x - self.xbar, self.A_nonlin(x).transpose(1,2)) + F.linear(u, self.B) + self.xbar
        assert f.shape==(batch_size, 1, self.num_states)
        return f

    def forward(self, t, x, u, w):
        # check sizes
        x = check_data_dim(x, vec_dim=(1, self.num_states))
        u = check_data_dim(u, vec_dim=(1, self.num_inputs))
        w = check_data_dim(w, vec_dim=(1, self.num_states))
        batch_size = x.shape[0]
        assert batch_size==u.shape[0], 'batch sizes of x and u are different.'
        assert batch_size==w.shape[0], 'batch sizes of w and u are different.'

        x_ = self.noiseless_forward(t, x, u) + w
        assert x_.shape==x.shape
        y = x_
        return x_, y

    # simulation
    def rollout(self, controller, data, train=False):
        """
        rollout REN for rollouts of the process noise
        - data: sequence of disturbance samples of shape
                (batch_size, T, num_states).
        """
        # check data shape to be (batch_size, T, num_states)
        data = to_tensor(data)
        if len(data.shape) == 1:
            data = torch.reshape(data, (-1, 1))
        data = check_data_dim(data, vec_dim=(None, self.num_states))
        batch_size, T = data.shape[0], data.shape[1]
        assert data.shape[-1] == self.num_states

        # init
        controller.reset()
        x = copy.deepcopy(self.x_init)
        x = x.reshape(1, *x.shape).repeat(batch_size, 1, 1)
        u = copy.deepcopy(self.u_init)
        u = u.reshape(1, *u.shape).repeat(batch_size, 1, 1)

        # Simulate
        for t in range(T):
            x, _ = self(t, x, u, data[:, t:t+1, :])
            assert x.shape==(batch_size, 1, self.num_states)
            u = controller(x)
            assert u.shape==(batch_size, 1, self.num_inputs)
            if t == 0:
                x_log, u_log = x, u
            else:
                x_log = torch.cat((x_log, x), 1)
                u_log = torch.cat((u_log, u), 1)
        assert x_log.shape==(batch_size, T, self.num_states)
        assert u_log.shape==(batch_size, T, self.num_inputs)

        controller.reset()
        if not train:
            x_log, u_log = x_log.detach(), u_log.detach()

        return x_log, None, u_log

