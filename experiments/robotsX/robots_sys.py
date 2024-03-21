import torch, copy
import torch.nn.functional as F

from config import device
from assistive_functions import to_tensor


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

        A1 = torch.eye(4*self.n_agents).to(device)
        A2 = torch.cat((torch.cat((torch.zeros(2,2),
                                   torch.eye(2)
                                   ), dim=1),
                        torch.cat((torch.diag(torch.tensor([-self.k/self.mass, -self.k/self.mass])),
                                   torch.diag(torch.tensor([-self.b/self.mass, -self.b/self.mass]))
                                   ),dim=1),
                        ),dim=0)
        A2 = torch.kron(torch.eye(self.n_agents), A2).to(device)
        self.A = A1 + self.h * A2

    def A_nonlin(self, x):
        if self.is_linear:
            raise NotImplementedError
        b2 = self.b2
        b1 = self.b
        m, k = self.mass, self.k
        A1 = torch.eye(4 * self.n_agents).to(device)
        A2 = torch.cat((torch.cat((torch.zeros(2, 2),
                                   torch.eye(2)
                                   ), dim=1),
                        torch.cat((torch.diag(torch.tensor([-k / m, -k / m])),
                                   torch.diag(torch.tensor([-b1 / m, -b1 / m]))
                                   ), dim=1),
                        ), dim=0)
        A2 = torch.kron(torch.eye(self.n_agents), A2).to(device)
        mask = torch.tensor([[0, 0], [1, 1]]).repeat(self.n_agents, 1).to(device)
        A3 = torch.norm(x.view(2 * self.n_agents, 2) * mask, dim=1, keepdim=True)
        A3 = torch.kron(A3, torch.ones(2, 1).to(device))
        A3 = -b2 / m * torch.diag(A3.squeeze()).to(device)
        A = A1 + self.h * (A2 + A3)
        return A

    def noiseless_forward(self, t, x, u):
        # NOTE: was called f
        if self.is_linear:
            f = F.linear(x - self.xbar, self.A) + F.linear(u, self.B) + self.xbar
        else:
            f = F.linear(x - self.xbar, self.A_nonlin(x)) + F.linear(u, self.B) + self.xbar
        return f

    def forward(self, t, x, u, w):
        x_ = self.noiseless_forward(t, x, u) + w
        y = x_
        return x_, y

    # simulation
    def multi_rollout(self, controller, data, train=False):
        """
        rollout REN for several rollouts of the process noise
        """
        data = to_tensor(data)
        assert len(data.shape) == 3
        num_rollouts = data.shape[0]
        for rollout_num in range(num_rollouts):
            x_log, _, u_log = self.rollout(
                controller, data=data[rollout_num, :, :], train=train
            )
            x_log = x_log.reshape(1, *x_log.shape)
            u_log = u_log.reshape(1, *u_log.shape)
            if rollout_num == 0:
                xs, us = x_log, u_log
            else:
                xs = torch.cat((xs, x_log), 0)
                us = torch.cat((us, u_log), 0)

        return xs, None, us

    def rollout(self, controller, data, train=False):
        """
        rollout REN for 1 rollout of the process noise
        """
        # check d is a 1D sequence of T disturbance samples
        data = to_tensor(data)
        if len(data.shape) == 1:
            data = torch.reshape(data, (-1, 1))
        assert len(data.shape) == 2
        T = data.shape[0]
        assert data.shape[1] == self.num_states

        # init
        controller.reset()
        x, u = copy.deepcopy(self.x_init), copy.deepcopy(self.u_init)

        # Simulate
        for t in range(T):
            x, _ = self(t, x, u, data[t, :])
            u = controller(x)
            if t == 0:
                x_log, u_log = x.reshape(1,-1), u.reshape(1,-1)
            else:
                x_log = torch.cat((x_log, x.reshape(1,-1)), 0)
                u_log = torch.cat((u_log, u.reshape(1,-1)), 0)

        controller.reset()
        if not train:
            x_log, u_log = x_log.detach(), u_log.detach()

        return x_log, None, u_log
