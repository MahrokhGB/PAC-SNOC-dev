import torch, pickle
import os, sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

random_seed = 5
torch.manual_seed(random_seed)

t_end = 100
std_ini = 0.2
n_agents = 2

file_path = os.path.join(BASE_DIR, 'experiments', 'robotsX', 'saved_results')
path_exist = os.path.exists(file_path)
if not path_exist:
    os.makedirs(file_path)
filename = 'data_T'+str(t_end)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'
filename = os.path.join(file_path, filename)

x0 = torch.tensor([2., -2, 0, 0,
                   -2, -2, 0, 0,
                   ])
xbar = torch.tensor([-2, 2, 0, 0,
                     2., 2, 0, 0,
                     ])

# train data
num_rollouts_big = 500      # generate 500 sequences, select as many as needed in the exp
num_states = 4*n_agents
train_data = torch.zeros(num_rollouts_big, t_end, num_states)
for rollout_num in range(num_rollouts_big):
    train_data[rollout_num, 0, :] = \
        (x0 - xbar) + std_ini * torch.randn(x0.shape)

# test data
num_rollouts_test = 500  # number of rollouts in the test data
test_data = torch.zeros(num_rollouts_test, t_end, num_states)
for rollout_num in range(num_rollouts_test):
    test_data[rollout_num, 0, :] = \
        (x0 - xbar) + std_ini * torch.randn(x0.shape)

data = {'train_data_full':train_data, 'test_data':test_data, 'x0':x0, 'xbar':xbar}

filehandler = open(filename, 'wb')
pickle.dump(data, filehandler)
filehandler.close()

