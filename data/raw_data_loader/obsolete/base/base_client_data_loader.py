import pickle
import h5py
import json
from abc import ABC, abstractmethod

from .utils import *


class BaseClientDataLoader(ABC):
    @abstractmethod
    def __init__(self, data_path, partition_path, client_idx, partition_method, tokenize, data_fields, clients_num):
        self.data_path = data_path
        self.partition_path = partition_path
        self.client_idx = client_idx
        self.partition_method = partition_method
        self.tokenize = tokenize
        self.data_fields = data_fields # ["x", "y"]
        self.clients_num = clients_num
        self.train_data = None
        self.test_data = None
        self.dev_data = None
        self.attributes = None
        self.load_data()
        if self.tokenize:
            self.spacy_tokenizer = SpacyTokenizer()

    def get_train_batch_data(self, batch_size=None):
        if batch_size is None:
            return self.train_data
        else:
            batch_data_list = list()
            start = 0
            length = len(self.train_data["Y"])
            while start < length:
                end = start + batch_size if start + batch_size < length else length
                batch_data = dict()
                for field in self.data_fields:
                    batch_data[field] = self.train_data[field][start: end]
                batch_data_list.append(batch_data)
                start = end
            return batch_data_list

    def get_test_batch_data(self, batch_size=None):
        if batch_size is None:
            return self.test_data
        else:
            batch_data_list = list()
            start = 0
            length = len(self.test_data["Y"])
            while start < length:
                end = start + batch_size if start + batch_size < length else length
                batch_data = dict()
                for field in self.data_fields:
                    batch_data[field] = self.test_data[field][start: end]
                batch_data_list.append(batch_data)
                start = end
            return batch_data_list

    def get_dev_batch_data(self, batch_size=None):
        if batch_size is None:
            return self.dev_data
        else:
            batch_data_list = list()
            start = 0
            length = len(self.dev_data["Y"])
            while start < length:
                end = start + batch_size if start + batch_size < length else length
                batch_data = dict()
                for field in self.data_fields:
                    batch_data[field] = self.dev_data[field][start: end]
                batch_data_list.append(batch_data)
                start = end
            return batch_data_list

    def get_train_data_num(self):
        if "X" in self.train_data:
            return len(self.train_data["X"])
        elif "context_X" in self.train_data:
            return len(self.train_data["context_X"])
        else:
            print(self.train_data.keys())
            return None
     
    def get_test_data_num(self):
        if "X" in self.test_data:
            return len(self.test_data["X"])
        elif "context_X" in self.test_data:
            return len(self.test_data["context_X"])
        else:
            return None

    def get_dev_data_num(self):
        if "X" in self.dev_data:
            return len(self.dev_data["X"])
        elif "context_X" in self.dev_data:
            return len(self.dev_data["context_X"])
        else:
            return None

    def get_attributes(self):
        return self.attributes

    def load_h5_data(self):
        data_dict = h5py.File(self.data_path, "r")
        partition_dict = h5py.File(self.partition_path, "r")

        n_clients = decode_data_from_h5(partition_dict[self.partition_method]["n_clients"][()])
        if n_clients < self.clients_num:
            raise ValueError(f"partition data have {n_clients} clients "
                             f"that mismatch you input {self.clients_num} clients")
        steps = int(n_clients / self.clients_num)

        def generate_client_data(data_dict, index_list):
            data = dict()
            for field in self.data_fields:
                data[field] = [decode_data_from_h5(data_dict[field][str(idx)][()]) for idx in index_list]
            return data

        if self.client_idx is None:
            train_index_list = []
            test_index_list = []
            for client_idx in partition_dict[self.partition_method]["partition_data"].keys():
                train_index_list.extend(decode_data_from_h5(partition_dict[self.partition_method]["partition_data"][client_idx]["train"][()]))
                test_index_list.extend(decode_data_from_h5(partition_dict[self.partition_method]["partition_data"][client_idx]["test"][()]))
            self.train_data = generate_client_data(data_dict, train_index_list)
            self.test_data = generate_client_data(data_dict, test_index_list)
        else:
            train_index_list, test_index_list = [], []
            start = max(self.client_idx*steps, 0)
            end = min((self.client_idx+1)*steps, n_clients)
            if start > end:
                raise ValueError(f"with {self.partition_method}, client number is more than")

            for client_idx in range(start, end):
                client_idx = str(client_idx)
                train_list = decode_data_from_h5(partition_dict[self.partition_method]["partition_data"][client_idx]["train"][()])
                train_index_list.extend(train_list)
                test_list = decode_data_from_h5(partition_dict[self.partition_method]["partition_data"][client_idx]["test"][()])
                test_index_list.extend(test_list)

            self.train_data = generate_client_data(data_dict, train_index_list)
            self.test_data = generate_client_data(data_dict, test_index_list)

        self.attributes = json.loads(data_dict["attributes"][()])
        self.attributes["n_clients"] = self.clients_num

        data_dict.close()
        partition_dict.close()

    def load_data(self):
        # data_dict = h5py.File(self.data_path, "r")
        # partition_dict = h5py.File(self.partition_path, "r")
        with open(self.data_path, "rb") as file:
            data_dict = pickle.load(file)
        with open(self.partition_path, "rb") as file:
            partition_dict = pickle.load(file)

        n_clients = partition_dict[self.partition_method]["n_clients"]
        if n_clients < self.clients_num:
            raise ValueError(f"partition data have {n_clients} clients "
                             f"that mismatch you input {self.clients_num} clients")

        steps = int(n_clients / self.clients_num)

        def generate_client_data(data_dict, index_list):
            data = dict()
            for field in self.data_fields:
                data[field] = [data_dict[field][str(idx)] for idx in index_list]
            return data

        if self.client_idx is None:
            train_index_list = []
            test_index_list = []
            dev_index_list = []
            for client_idx in partition_dict[self.partition_method]["partition_data"].keys():
                train_index_list.extend(partition_dict[self.partition_method]["partition_data"][client_idx]["train"])
                dev_index_list.extend(partition_dict[self.partition_method]["partition_data"][client_idx]["valid"])
                test_index_list.extend(partition_dict[self.partition_method]["partition_data"][client_idx]["test"])
            self.train_data = generate_client_data(data_dict, train_index_list)
            self.test_data = generate_client_data(data_dict, test_index_list)
            self.dev_data = generate_client_data(data_dict, dev_index_list)
        else:
            train_index_list, test_index_list, dev_index_list = [], [], []
            start = max(self.client_idx*steps, 0)
            end = min((self.client_idx+1)*steps, n_clients)
            if start > end:
                raise ValueError(f"with {self.partition_method}, client number is more than")

            for client_idx in range(start, end):
                client_idx = str(client_idx)
                train_list = partition_dict[self.partition_method]["partition_data"][client_idx]["train"]
                train_index_list.extend(train_list)
                test_list = partition_dict[self.partition_method]["partition_data"][client_idx]["test"]
                test_index_list.extend(test_list)
                dev_list = partition_dict[self.partition_method]["partition_data"][client_idx]["valid"]
                dev_index_list.extend(dev_list)

            self.train_data = generate_client_data(data_dict, train_index_list)
            self.test_data = generate_client_data(data_dict, test_index_list)
            self.dev_data = generate_client_data(data_dict, dev_index_list)

        self.attributes = data_dict["attributes"]
        self.attributes["n_clients"] = self.clients_num