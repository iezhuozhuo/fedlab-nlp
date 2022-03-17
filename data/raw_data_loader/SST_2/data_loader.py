import os
import pandas as pd
from nltk.tree import Tree

from data.raw_data_loader.base.base_raw_data_loader import TextClassificationRawDataLoader


class RawDataLoader(TextClassificationRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_file_name = "train.tsv"
        self.test_file_name = "dev.tsv"
        self.load_data()

    def load_data(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.attributes["label_vocab"] is None:
            train_size = self.process_data_file(os.path.join(self.data_path, self.train_file_name))
            test_size = self.process_data_file(os.path.join(self.data_path, self.test_file_name))
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [i for i in range(train_size, train_size + test_size)]
            self.attributes["index_list"] = self.attributes["train_index_list"] + self.attributes["test_index_list"]
            self.attributes["label_vocab"] = {label: i for i, label in enumerate(set(self.Y.values()))}

    def label_level(self, label):
        return {'0': 'negative', '1': 'negative', '2': 'neutral',
                    '3': 'positive', '4': 'positive', None: None}[label]

    #
    def process_data_file(self, file_path):
        cnt = 0
        with open(file_path, "r") as f:
            for i,line in enumerate(f):
                if i == 0:
                    continue
                text, label = line.strip().split("\t")
                idx = len(self.X)
                self.X[idx], self.Y[idx] = text, label
                cnt += 1
        print(f"load {cnt} from {file_path}")
                # tree = Tree.fromstring(line)
                # label = self.label_level(tree.label())
                # if label != "neutral":
                #     assert len(self.X) == len(self.Y)
                #     idx = len(self.X)
                #     self.X[idx] = " ".join(tree.leaves())
                #     self.Y[idx] = label
                #     cnt += 1
        return cnt

