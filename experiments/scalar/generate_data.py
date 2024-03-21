"""
script to generate data
1024 train and 1024 test samples are saved.
The entire test data is used for testing all methods.
A subset of the training dataset is used for training ours and empirical.
The benchmark is trained on the entire data
"""
import pickle
import numpy as np
import os, sys


BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(1, BASE_DIR)


def generate_data_func(S, T, random_state, disturbances, num_states):
    data = dict.fromkeys(disturbances.keys())
    for dist_type in disturbances.keys():
        d = disturbances[dist_type]
        if dist_type in ['N 0-mean', 'N biased']:
            data[dist_type] = random_state.multivariate_normal(
                mean=d['mean'], cov=d['cov'], size=(S,T)
            )
        elif dist_type == 'N multi-modal':
            # A stream of indices from which to choose the component
            mixture_idx = random_state.choice(
                len(d['weight']), size=S*T, replace=True, p=d['weight']
            )
            # y is the mixture sample
            data[dist_type] = np.array([
                    [random_state.multivariate_normal(
                        d['mean'][mixture_idx[s_ind*S + t_ind]], d['cov']
                    ) for t_ind in range(T)]
                for s_ind in range(S)]
            )
        elif dist_type == 'Uniform':
            data[dist_type] = random_state.uniform(
                low=d['low'], high=d['high'], size=(S,T)
            )
        else:
            raise NotImplementedError
        data[dist_type] = np.reshape(
            data[dist_type],
            (S, T, num_states)
        )
    return data


if __name__ == '__main__':
    random_seed = 33
    T = 10
    dist_type = 'N biased'
    file_path = os.path.join(BASE_DIR, 'experiments', 'scalar', 'saved_results')
    path_exist = os.path.exists(file_path)
    if not path_exist:
        os.makedirs(file_path)
    filename = dist_type.replace(" ", "_")+'_data_T'+str(T)+'_RS'+str(random_seed)+'.pkl'
    filename = os.path.join(file_path, filename)

    num_states = 1
    d_dist_v = 0.3*np.ones((num_states, 1))
    d_dist_cov = np.matmul(d_dist_v, np.transpose(d_dist_v))  # used in all
    disturbances = {
        'N 0-mean':{'mean':np.zeros(num_states), 'cov':d_dist_cov},
        'N 0-mean trunc':{
            'mean':np.zeros(num_states), 'cov':d_dist_cov,
            'lb':-0.7*np.ones(num_states), 'ub':0.7*np.ones(num_states)},
        'N biased':{'mean':0.3*np.ones(num_states), 'cov':d_dist_cov},
        # 'N multi-modal':{'mean':, 'cov':, 'weights':None, 'lb':None, 'ub':None}
    }

    random_state = np.random.RandomState(random_seed)
    data_tr = generate_data_func(
        1024, T, random_state,
        {dist_type: disturbances[dist_type]}, num_states
    )
    data_ts = generate_data_func(
        1024, T, random_state,
        {dist_type: disturbances[dist_type]}, num_states
    )
    data_all = {'train_big':data_tr, 'test_big':data_ts, 'disturbance':disturbances[dist_type]}
    filehandler = open(filename, 'wb')
    pickle.dump(data_all, filehandler)
    print("Data saved at " + filename)
    filehandler.close()
