import os
import json
from collections import defaultdict
from abc import ABC, abstractmethod

from data.raw_data_loader.base.partition import homo_partition


class BaseRawDataLoader(ABC):
    @abstractmethod
    def __init__(self, data_path, client_num, partition_file_path, partition_type):
        if os.path.isdir(data_path):
            data_file = os.path.join(data_path, "data.json")
        else:
            data_file = data_path
            data_path = "/".join(data_path.split("/")[0:-1])

        self.data_path = data_path
        self.data_file = data_file
        self.client_num = client_num
        self.partition_file_path = partition_file_path
        self.partition_type = partition_type
        self.attribute = {"n_clients": client_num}
        # self.attributes["index_list"] = None

    @abstractmethod
    def load_raw_data(self):
        pass

    @abstractmethod
    def process_data_file(self, file_path):
        pass

    @abstractmethod
    def generate_partition_file(self):
        pass


class TextClassificationRawDataLoader(BaseRawDataLoader):
    def __init__(self, data_path, client_num, partition_file_path, partition_type):
        super(TextClassificationRawDataLoader, self).__init__(data_path, client_num, partition_file_path,
                                                              partition_type)
        self.X = defaultdict(dict)
        self.Y = defaultdict(dict)
        self.partition_dict = None
        # self.attributes num_labels/label_vocab/task_type/train_num/test_num/n_clients
        self.attribute["num_labels"] = -1
        self.attribute["label_vocab"] = None
        self.attribute["task_type"] = "text_classification"

    def generate_partition_file(self):
        if os.path.exists(self.partition_file_path):
            with open(self.partition_file_path) as file:
                self.partition_dict = json.load(file)
        else:
            if self.partition_type == "uniform":
                train_partition_dict = homo_partition(self.attribute["train_num"], self.client_num)
                test_partition_dict = homo_partition(self.attribute["test_num"], self.client_num)
            else:
                train_partition_dict, test_partition_dict = dict(), dict()

            self.partition_dict = {"train": train_partition_dict, "test": test_partition_dict}
            with open(self.partition_file_path, "w") as file:
                json.dump(self.partition_dict, file)