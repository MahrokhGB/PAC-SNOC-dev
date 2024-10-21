import torch, os, pickle
from torch.utils.data import Dataset

from config import BASE_DIR, device

class RobotsDataset(Dataset):
    def __init__(self, random_seed, horizon, std_ini=0.2, n_agents=2):
        self.random_seed = random_seed
        self.horizon = horizon
        self.std_ini = std_ini
        self.n_agents = n_agents

        # initial state TODO: set as arg
        self.x0 = torch.tensor([2., -2, 0, 0,
                        -2, -2, 0, 0,
                        ], device=device)
        self.xbar = torch.tensor([-2, 2, 0, 0,
                            2., 2, 0, 0,
                        ], device=device)

        # file name and path
        file_path = os.path.join(BASE_DIR, 'experiments', 'minimal_example', 'saved_results')
        path_exist = os.path.exists(file_path)
        if not path_exist:
            os.makedirs(file_path)
        filename = 'data_T'+str(self.horizon)+'_stdini'+str(self.std_ini)+'_agents'+str(self.n_agents)+'_RS'+str(self.random_seed)+'.pkl'
        self.filename = os.path.join(file_path, filename)

        # load data if generated or generate if doesn't exist
        self._load_data()

    def __len__(self):
        return len(self.data.shape[0])

    def __getitem__(self, idx):
        return self.data[idx, :, :]

    # ---- data generation ----
    def _generate_data(self):
        torch.manual_seed(self.random_seed)

        # train data
        num_rollouts_big = 500      # generate 500 sequences, select as many as needed in the exp
        num_states = 4*self.n_agents
        self.train_data_full = torch.zeros(num_rollouts_big, self.horizon, num_states, device=device)
        for rollout_num in range(num_rollouts_big):
            self.train_data_full[rollout_num, 0, :] = \
                (self.x0 - self.xbar) + self.std_ini * torch.randn(self.x0.shape).to(device)

        # test data
        num_rollouts_test = 500  # number of rollouts in the test data
        self.test_data = torch.zeros(num_rollouts_test, self.horizon, num_states, device=device)
        for rollout_num in range(num_rollouts_test):
            self.test_data[rollout_num, 0, :] = \
                (self.x0 - self.xbar) + self.std_ini * torch.randn(self.x0.shape).to(device)

        # save
        filehandler = open(self.filename, 'wb')
        pickle.dump({'train_data_full':self.train_data_full, 'test_data':self.test_data}, filehandler)
        filehandler.close()

    # ---- load data ----
    def _load_data(self):
        # check if data exists
        if os.path.isfile(self.filename):
            filehandler = open(self.filename, 'rb')
            data = pickle.load(filehandler)
            self.train_data_full = data['train_data_full']
            self.test_data = data['test_data']
            filehandler.close()
        else:
            self._generate_data()
