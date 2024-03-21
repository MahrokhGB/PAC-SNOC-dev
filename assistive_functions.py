import numpy as np
import torch
from config import device
from matplotlib.patches import Rectangle


def to_tensor(x):
    return torch.from_numpy(x).contiguous().float().to(device) if isinstance(x, np.ndarray) else x


class DummyLRScheduler:

    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass


class WrapLogger():
    def __init__(self, logger, verbose=True):
        self.can_log = not (logger == None)
        self.logger=logger
        self.verbose = verbose

    def info(self, msg):
        if self.can_log:
            self.logger.info(msg)
        if self.verbose:
            print(msg)

    def close(self):
        if not self.can_log:
            return
        while len(self.logger.handlers):
            h = self.logger.handlers[0]
            h.close()
            self.logger.removeHandler(h)


def plot_rect(theta, bias, theta_grid, bias_grid, ax, label, color):
    theta_min_ind = np.argmin(abs(theta_grid-theta))
    bias_min_ind = np.argmin(abs(bias_grid-bias))
    r_ind_theta = len(theta_grid)-1-theta_min_ind
    r_ind_bias = len(bias_grid)-1-bias_min_ind
    ax.add_patch(
        Rectangle((r_ind_bias, r_ind_theta),
        1, 1, fill=False, edgecolor=color, lw=3, label=label, linestyle='-'))
    return