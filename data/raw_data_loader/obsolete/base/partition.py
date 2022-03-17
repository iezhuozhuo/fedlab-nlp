import math
import random
import numpy as np

from ..base.globals import *


def uniform_partition(train_index_list, test_index_list=None, n_clients=N_CLIENTS):
    partition_dict = dict()
    partition_dict["n_clients"] = n_clients
    partition_dict["partition_data"] = dict()
    train_index_list = train_index_list.copy()
    random.shuffle(train_index_list)
    train_batch_size = math.ceil(len(train_index_list) / n_clients)

    test_batch_size = None

    if test_index_list is not None:
        test_index_list = test_index_list.copy()
        random.shuffle(test_index_list)
        test_batch_size = math.ceil(len(test_index_list) / n_clients)
    for i in range(n_clients):
        train_start = i * train_batch_size
        partition_dict["partition_data"][i] = dict()
        train_set = train_index_list[train_start: train_start + train_batch_size]
        if test_index_list is None:
            random.shuffle(train_set)
            train_num = int(len(train_set) * 0.8)
            partition_dict["partition_data"][i]["train"] = train_set[:train_num]
            partition_dict["partition_data"][i]["test"] = train_set[train_num:]
        else:
            test_start = i * test_batch_size
            partition_dict["partition_data"][i]["train"] = train_set
            partition_dict["partition_data"][i]["test"] = test_index_list[test_start:test_start + test_batch_size]

    return partition_dict


def split_indices(num_cumsum, rand_perm):
    client_indices_pairs = [(cid, idxs) for cid, idxs in
                            enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    client_dict = dict(client_indices_pairs)
    return client_dict


def homo_partition(sample_num, client_num):
    """Partition data indices in IID way given sample numbers for each clients.
    Args:
        client_num (int): Number of clients.
        sample_num (int): Number of samples.
    Returns:
        dict: ``{ client_id: indices}``.
    """
    client_sample_nums = average_sample_nums(sample_num, client_num)

    rand_perm = np.random.permutation(sample_num)
    num_cumsum = np.cumsum(client_sample_nums).astype(int)
    client_dict = split_indices(num_cumsum, rand_perm)
    return {key: list(value.astype(str)) for key, value in client_dict.items()}