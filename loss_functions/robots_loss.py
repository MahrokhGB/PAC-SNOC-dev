import torch
import torch.nn.functional as F
from config import device
from assistive_functions import to_tensor
from . import LQLossFH

class LossRobots(LQLossFH):
    def __init__(
        self, T, xbar, loss_bound, sat_bound,
        n_agents, Q, alpha_u=1,
        alpha_ca=None, alpha_obst=None, min_dist=None
    ):
        super().__init__(Q=Q, R=alpha_u, T=T, loss_bound=loss_bound, sat_bound=sat_bound)
        self.xbar = xbar # TODO
        self.alpha_u = alpha_u # TODO
        self.n_agents = n_agents
        self.alpha_ca, self.alpha_obst, self.min_dist = alpha_ca, alpha_obst, min_dist

        assert (self.alpha_ca is None and self.min_dist is None) or not (self.alpha_ca is None or self.min_dist is None)
        if not self.alpha_ca is None:
            assert not self.n_agents is None

    def forward(self, x_log, u_log):
        (num_rollouts, T, _) = x_log.shape
        assert T == self.T and u_log.shape[1] == self.T and u_log.shape[0] == num_rollouts
        loss_tot = 0
        for rollout_num in range(num_rollouts):
            loss_x, loss_u, loss_ca, loss_obst = 0, 0, 0, 0
            for t in range(self.T):
                loss_x += f_loss_states(
                    t, x=x_log[rollout_num, t, :], xbar=self.xbar, Q=self.Q
                )
                loss_u += self.alpha_u * f_loss_u(t, u_log[rollout_num, t, :])
                if not self.alpha_ca is None:
                    loss_ca += self.alpha_ca * f_loss_ca(
                        x=x_log[rollout_num, t, :], n_agents=self.n_agents,
                        num_states=self.num_states, min_dist=self.min_dist
                    )
                if not self.alpha_obst is None:
                    loss_obst += self.alpha_obst * f_loss_obst(x_log[rollout_num, t, :])
            # divide by time horizon
            loss_x, loss_u = loss_x/T, loss_u/T
            loss_ca, loss_obst = loss_ca/T, loss_obst/T
            # add
            loss_val = loss_x + loss_u + loss_ca + loss_obst
            # bound
            if self.sat_bound is not None:
                # LQ loss>=0, so, tanh(LQ loss) in [0,1]
                loss_val = torch.tanh(loss_val/self.sat_bound)
            if self.loss_bound is not None:
                loss_val = self.loss_bound * loss_val
            # add to total loss
            loss_tot += loss_val
        loss_tot = loss_tot/num_rollouts
        return loss_tot


def f_loss_ca(x, n_agents, num_states, min_dist=0.5):
    min_sec_dist = min_dist + 0.2
    # collision avoidance:
    deltaqx = x[0::4].repeat(n_agents, 1) - x[0::4].repeat(n_agents, 1).transpose(0, 1)
    deltaqy = x[1::4].repeat(n_agents, 1) - x[1::4].repeat(n_agents, 1).transpose(0, 1)
    distance_sq = deltaqx ** 2 + deltaqy ** 2
    mask = torch.logical_not(torch.eye(num_states // 4).to(device))
    loss_ca = (1/(distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2)) * mask).sum()/2
    return loss_ca


def normpdf(q, mu, cov):
    d = 2
    mu = mu.view(1, d)
    cov = cov.view(1, d)  # the diagonal of the covariance matrix
    qs = torch.split(q, 2)
    out = torch.tensor(0).to(device)
    for qi in qs:
        # if qi[1]<1.5 and qi[1]>-1.5:
        den = (2*torch.pi)**(0.5*d) * torch.sqrt(torch.prod(cov))
        num = torch.exp((-0.5 * (qi - mu)**2 / cov).sum())
        out = out + num/den
    return out


def f_loss_obst(x, sys=None):
    qx = x[::4].unsqueeze(1)
    qy = x[1::4].unsqueeze(1)
    q = torch.cat((qx,qy), dim=1).view(1,-1).squeeze()
    mu1 = torch.tensor([[-2.5, 0]]).to(device)
    mu2 = torch.tensor([[2.5, 0.0]]).to(device)
    mu3 = torch.tensor([[-1.5, 0.0]]).to(device)
    mu4 = torch.tensor([[1.5, 0.0]]).to(device)
    cov = torch.tensor([[0.2, 0.2]]).to(device)
    Q1 = normpdf(q, mu=mu1, cov=cov)
    Q2 = normpdf(q, mu=mu2, cov=cov)
    Q3 = normpdf(q, mu=mu3, cov=cov)
    Q4 = normpdf(q, mu=mu4, cov=cov)

    return (Q1 + Q2 + Q3 + Q4).sum()


def f_loss_side(x):
    qx = x[::4]
    qy = x[1::4]
    side = torch.relu(qx - 3) + torch.relu(-3 - qx) + torch.relu(qy - 6) + torch.relu(-6 - qy)
    return side.sum()
