import pickle, sys, os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(1, BASE_DIR)
from config import device
from assistive_functions import WrapLogger
from loss_functions import LQLossFH
from controllers.empirical_controller import *
from experiments.scalar.LTI_sys import LTI_system
from experiments.scalar.scalar_assistive_functions import load_data

random_seed = 33
random_state = np.random.RandomState(random_seed)
logger = WrapLogger(None)


# ------ 1. load data ------
T = 10
dist_type = 'N biased'
S = None# use all training data
data_train, data_test, disturbance = load_data(
    dist_type=dist_type, S=S, T=T, random_seed=random_seed,
    S_test=None   # use a subset of available test data if not None
)

# ------ 2. define the plant ------
sys_np = LTI_system(
    A=np.array([[0.8]]),  # num_states*num_states
    B=np.array([[0.1]]),  # num_states*num_inputs
    C=np.array([[0.3]]),  # num_outputs*num_states
    x_init=2*np.ones((1, 1)),  # num_states*1
    use_tensor=False
)
sys = LTI_system(
    sys_np.A, sys_np.B, sys_np.C, sys_np.x_init,
    use_tensor=True
)

# ------ 3. define the loss ------
Q = 5*torch.eye(sys.num_states).to(device)
R = 0.003*torch.eye(sys.num_inputs).to(device)
# optimal loss bound
loss_bound = 1
# sat_bound = np.matmul(np.matmul(np.transpose(sys.x_init), Q) , sys.x_init)
sat_bound = torch.matmul(torch.matmul(torch.transpose(sys.x_init, 0, 1), Q), sys.x_init)
if loss_bound is not None:
    logger.info('[INFO] bounding the loss to ' + str(loss_bound))
lq_loss_bounded = LQLossFH(Q, R, T, loss_bound, sat_bound, logger=logger)
lq_loss_original = LQLossFH(Q, R, T, None, None, logger=logger)

# ------ 4. Benchmark ------
batch_size_bnc = 20
controller_bnc = EmpCont(
    sys=sys, train_d=data_train, lr=0.5, loss=lq_loss_original,
    random_seed=random_seed, optimizer='AdamW',
    requires_bias={'hidden':False, 'out':False}  # bias is set to the true mean, not learned
)
controller_bnc.controller.out.bias = to_tensor(
    -disturbance['mean']/sys_np.B[0,0]
)   # set bias according to the true unknown noise distribution
# fit
controller_bnc.fit(num_iter_fit=25000, batch_size=batch_size_bnc, log_period=1000)

# save
filename = dist_type.replace(" ", "_")+'_bnc_T'+str(T)+'.pkl'
filename = os.path.join(file_path, filename)
filehandler = open(filename, 'wb')
dict_cont = {
    'weight':controller_bnc.controller.out.weight.detach().cpu(),
    'bias':controller_bnc.controller.out.bias.detach().cpu()
}
pickle.dump(dict_cont, filehandler)
filehandler.close()
logger.info('[INFO] benchmark controller with T='+str(T)+' using random seed'+str(random_seed)+' saved.')
logger.info(
    '----- Benchmark gain = {:.3f}, bias = {:.3f} -----\n'.format(
        controller_bnc.controller.out.weight.item(),
        controller_bnc.controller.out.bias.item()
    )
)
