import torch, time

from controllers.abstract import CLSystem, LinearController
from assistive_functions import to_tensor


# This class `EmpCont` is used for fitting a controller to a system
# using PyTorch through minimizing the empirical cose. Options are
# available for different optimizers and training methods.
class EmpCont(CLSystem):
    def __init__(
        self, sys, train_d, lr, loss,
        init_weight=None, init_bias=None,
        random_seed=None, optimizer='SGD',
        requires_bias={'out':True, 'hidden':True}
    ):
        """
        train_d: (ndarray) train disturbances - shape: (S, T, sys.num_states)
        """
        # --- init ---
        lin_controller = LinearController(
            sys.num_states, sys.num_inputs, requires_bias
        )
        super().__init__(sys, lin_controller, random_seed)
        (self.S, self.T, num_states) = train_d.shape
        assert num_states == self.sys.num_states
        assert self.sys.use_tensor, "the system should be defined to use tensors"

        # --- convert the data into pytorch tensors ---
        self.train_d = to_tensor(train_d)

        # if not (init_weight is None and init_bias is None):
        #     lin_controller.set_params(init_weight, init_bias)

        # --- setup optimizer ---
        self.parameters = [{'params': self.controller.parameters(), 'lr': lr}]
        self.fitted = False
        self.loss = loss
        if optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.parameters, lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters, lr=lr)
        else:
            raise NotImplementedError('Optimizer must be AdamW or SGD')

    def fit(self, num_iter_fit, batch_size, log_period=500):
        """
        fits controller parameters of by minimizing the empirical training cost

        Args:
            num_iter_fit: number of iterations
            batch_size: number of sequences used in each iteration
            log_period: (int) number of steps after which to print stats
        """
        if batch_size > self.train_d.shape[0]:
            print('[WARNING] batch size is larger than the total data size. set to data size')
            batch_size = self.train_d.shape[0]

        self.train()

        t = time.time()
        for itr in range(1, num_iter_fit + 1):
            # sample data batch
            inds = self.random_state.permutation(self.train_d.shape[0])[:batch_size]

            # take a step
            self.optimizer.zero_grad()
            xs, _, us = self.multi_rollout(self.train_d[inds, :])
            loss = self.loss.forward(xs, us)
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # print training stats
            if itr == 1 or itr % log_period == 0:
                duration = time.time() - t
                t = time.time()
                message = '\nIter %d/%d - Loss: %.3f - Time %.3f sec' % (itr, num_iter_fit, loss.item(), duration)
                print(message)
                # self.controller.print_params()

        # end of training
        self.fitted = True
        self.eval()
        return loss.item()

    def forward(self, x):
        return self.controller.forward(x)

