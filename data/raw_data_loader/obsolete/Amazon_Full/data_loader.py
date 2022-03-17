import os
from tqdm import tqdm

from ..base.utils import *
from ..base.my_raw_data_loader import TextClassificationRawDataLoader

from training.utils.register import registry


class AmazonRawDataLoader(TextClassificationRawDataLoader):
    def __init__(self, data_path, client_num, partition_file_path, partition_type):
        super().__init__(data_path, client_num, partition_file_path, partition_type)
        self.train_path = "train.csv"
        self.test_path = "test.csv"

        self.load_raw_data()
        self.generate_partition_file()

    def load_raw_data(self):
        if not os.path.exists(self.data_file):
            self.attribute["train_num"] = self.process_data_file(os.path.join(self.data_path, self.train_path), "train")
            self.attribute["test_num"] = self.process_data_file(os.path.join(self.data_path, self.test_path), "test")
            self.attribute["label_vocab"] = {label: i for i, label in enumerate(set(
                list(self.Y["train"].values()) + list(self.Y["test"].values())))}
            self.attribute["num_labels"] = len(self.attribute["label_vocab"])
            with open(self.data_file, "w") as file:
                data_dict = {"X": self.X, "Y": self.Y, "attributes": self.attribute}
                json.dump(data_dict, file)
        else:
            with open(self.data_file) as file:
                data_dict = json.load(file)
                self.X = data_dict["X"]
                self.Y = data_dict["Y"]
                self.attribute = data_dict["attributes"]

    def process_data_file(self, file_path, data_type="train"):
        with open(file_path) as file:
            cnt = 0
            for i, line in enumerate(file):
                line_list = line.strip().split(",")
                label, title, review = line_list[0], line_list[1], ",".join(line_list[2:])
                label, title, review = [s.replace('"', "") for s in [label, title, review]]
                if not label.isnumeric():
                    continue
                context = ".".join([review, title])
                assert len(self.X) == len(self.Y)
                idx = len(self.X[data_type])
                self.X[data_type][idx] = context
                self.Y[data_type][idx] = label
                cnt += 1
        return cnt


class ClientDataLoader(object):
    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False,
                 clients_num=100):
        data_fields = ["X", "Y"]
        self.logger = registry.get("logger")
        self.data_path = data_path
        self.partition_path = partition_path
        self.client_idx = client_idx
        self.partition_method = partition_method
        self.tokenize = tokenize
        self.data_fields = data_fields  # ["x", "y"]
        self.clients_num = clients_num
        self.train_data = dict()
        self.test_data = dict()
        self.attribute = None
        self.partition_dict = None

        self.load_data()

        if self.tokenize:
            if os.path.isdir(self.data_path):
                path = self.data_path
            else:
                path = "/".join(self.data_path.split("/")[0:-1])
            self.tokenized_data_path = os.path.join(path, "tokenized_data.json")
            self.spacy_tokenizer = SpacyTokenizer()
            self.tokenize_data()

    def tokenize_data(self):
        if os.path.exists(self.tokenized_data_path):
            self.logger.info(f"Loading tokenized data from {self.tokenized_data_path}")
            with open(self.tokenized_data_path) as file:
                tokenized_data = json.load(file)
            self.train_data = tokenized_data["train_data"]
            self.test_data = tokenized_data["test_data"]
        else:
            self.logger.info(f"generating tokenized data from {self.tokenized_data_path}")
            tokenizer = self.spacy_tokenizer.en_tokenizer

            def __tokenize_data(data):
                for i in tqdm(range(len(data["X"])), desc="Tokenize"):
                    data["X"][i] = [token.text.strip().lower() for token in tokenizer(data["X"][i].strip()) if
                                    token.text.strip()]

            __tokenize_data(self.train_data)
            __tokenize_data(self.test_data)

            with open(self.tokenized_data_path, "w") as file:
                tokenized_data = {"train_data": self.train_data,
                                  "test_data": self.test_data}
                json.dump(tokenized_data, file)

    def load_data(self):
        self.logger.info(f"Loading data from {self.data_path}")
        row_data_loader = AmazonRawDataLoader(
            data_path=self.data_path, client_num=self.clients_num,
            partition_file_path=self.partition_path, partition_type=self.partition_method)
        self.attribute = row_data_loader.attribute
        self.partition_dict = row_data_loader.partition_dict

        if not self.client_idx:
            self.train_data["X"] = [row_data_loader.X["train"][index] for index in row_data_loader.X["train"]]
            self.train_data["Y"] = [row_data_loader.Y["train"][index] for index in row_data_loader.Y["train"]]
            self.test_data["X"] = [row_data_loader.X["test"][index] for index in row_data_loader.X["test"]]
            self.test_data["Y"] = [row_data_loader.Y["test"][index] for index in row_data_loader.Y["test"]]
        else:
            train_index_list = self.partition_dict["train"][self.client_idx]
            test_index_list = self.partition_dict["test"][self.client_idx]
            self.train_data["X"] = [row_data_loader.X["train"][int(index)] for index in train_index_list]
            self.train_data["Y"] = [row_data_loader.Y["train"][int(index)] for index in test_index_list]

            self.test_data["X"] = [row_data_loader.X["test"][int(index)] for index in test_index_list]
            self.test_data["Y"] = [row_data_loader.Y["test"][int(index)] for index in test_index_list]

    def get_train_batch_data(self, batch_size=None):
        return self.get_batch_data(batch_size, "train")

    def get_test_batch_data(self, batch_size=None):
        return self.get_batch_data(batch_size, "test")

    def get_batch_data(self, batch_size=None, data_type="train"):
        data = self.train_data if data_type == "train" else self.test_data
        if not batch_size:
            return data
        else:
            batch_data_list = list()
            start = 0
            length = len(data["Y"])
            while start < length:
                end = start + batch_size if start + batch_size < length else length
                batch_data = dict()
                for field in self.data_fields:
                    batch_data[field] = data[field][start: end]
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

    def get_attributes(self):
        return self.attribute
