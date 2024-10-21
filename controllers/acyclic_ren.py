import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from config import device

class AcyclicREN(nn.Module):
    """
    Acyclic recurrent equilibrium network, following the paper:
    "Recurrent equilibrium networks: Flexible dynamic models with guaranteed
    stability and robustness, Revay M et al. ."

    The mathematical model of RENs relies on an implicit layer embedded in a recurrent layer.
    The model is described as,

                    [  E . x_t+1 ]  =  [ F    B_1  B_2   ]   [  x_t ]   +   [  b_x ]
                    [  Λ . v_t   ]  =  [ C_1  D_11  D_12 ]   [  w_t ]   +   [  b_w ]
                    [  y_t       ]  =  [ C_2  D_21  D_22 ]   [  u_t ]   +   [  b_u ]

    where E is an invertible matrix and Λ is a positive-definite diagonal matrix. The model parameters
    are then {E, Λ , F, B_i, C_i, D_ij, b} which form a convex set according to the paper.

    NOTE: REN has input "u", output "y", and internal state "x". When used in closed-loop,
          the REN input "u" would be the noise reconstruction ("w") and the REN output ("y")
          would be the input to the plant. The internal state of the REN should not be mistaken
          with the internal state of the plant.
    """

    def __init__(
        self, dim_in: int, dim_out: int, dim_internal: int,
        l: int, internal_state_init = None, initialization_std: float = 0.5,
        posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0
    ):
        """
        Args:
            dim_in (int): Input (u) dimension.
            dim_out (int): Output (y) dimension.
            dim_internal (int): Internal state (x) dimension. This state evolves with contraction properties.
            l (int): Complexity of the implicit layer.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            internal_state_init (torch.Tensor or None, optional): Initial condition for the internal state. Defaults to 0 when set to None.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
        """
        super().__init__()

        # set dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_internal = dim_internal
        self.l = l

        # set functionalities
        self.contraction_rate_lb = contraction_rate_lb

        # initialize internal state
        if internal_state_init is None:
            self.x = torch.zeros(1, 1, self.dim_internal, device=device)
        else:
            assert isinstance(internal_state_init, torch.Tensor)
            self.x = internal_state_init.reshape(1, 1, self.dim_internal).to(device)

        # define matrices shapes
        # auxiliary matrices
        self.X_shape = (2 * self.dim_internal + self.l, 2 * self.dim_internal + self.l)
        self.Y_shape = (self.dim_internal, self.dim_internal)
        # nn state dynamics
        self.B2_shape = (self.dim_internal, self.dim_in)
        # nn output
        self.C2_shape = (self.dim_out, self.dim_internal)
        self.D21_shape = (self.dim_out, self.l)
        self.D22_shape = (self.dim_out, self.dim_in)
        # v signal
        self.D12_shape = (self.l, self.dim_in)

        # define trainble params
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
        for name in self.training_param_names:
            # read the defined shapes
            shape = getattr(self, name + '_shape')
            # define each param as nn.Parameter
            setattr(self, name, nn.Parameter((torch.randn(*shape, device=device) * initialization_std)))

        # auxiliary elements
        self.epsilon = posdef_tol

        # update non-trainable model params
        self.update_model_param()

    def update_model_param(self):
        """
        Update non-trainable matrices according to the REN formulation to preserve contraction.
        """
        # dependent params
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * self.dim_internal + self.l, device=device)
        h1, h2, h3 = torch.split(H, [self.dim_internal, self.l, self.dim_internal], dim=0)
        H11, H12, H13 = torch.split(h1, [self.dim_internal, self.l, self.dim_internal], dim=1)
        H21, H22, _ = torch.split(h2, [self.dim_internal, self.l, self.dim_internal], dim=1)
        H31, H32, H33 = torch.split(h3, [self.dim_internal, self.l, self.dim_internal], dim=1)
        P = H33

        # nn state dynamics
        self.F = H31
        self.B1 = H32

        # nn output
        self.E = 0.5 * (H11 + self.contraction_rate_lb * P + self.Y - self.Y.T)

        # v signal for strictly acyclic REN
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, u_in):
        """
        Forward pass of REN.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """
        batch_size = u_in.shape[0]

        w = torch.zeros(batch_size, 1, self.l, device=device)

        # update each row of w using Eq. (8) with a lower triangular D11
        eye_mat = torch.eye(self.l, device=device)
        for i in range(self.l):
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u_in, self.D12[i,:])
            w = w + (eye_mat[i, :] * torch.tanh(v / self.Lambda[i])).reshape(batch_size, 1, self.l)

        # compute next state using Eq. 18
        self.x = F.linear(
            F.linear(self.x, self.F) + F.linear(w, self.B1) + F.linear(u_in, self.B2),
            self.E.inverse())

        # compute output
        y_out = F.linear(self.x, self.C2) + F.linear(w, self.D21) + F.linear(u_in, self.D22)
        return y_out

    # setters and getters
    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in self.training_param_names
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in self.training_param_names
        )
        return param_dict

