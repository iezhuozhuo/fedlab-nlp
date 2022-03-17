from data_manager.base_data_manager import BaseDataManager
from torch.utils.data import DataLoader
import h5py
import json
import logging
from tqdm import tqdm


class TextClassificationDataManager(BaseDataManager):
    """Data manager for text classification"""
    def __init__(self, args, model_args, preprocessor, rank, client_index_list):
        super(TextClassificationDataManager, self).__init__(args, model_args, rank, client_index_list)
        self.attributes = self.load_attributes(args.data_file_path)
        self.preprocessor = preprocessor

    def read_instance_from_data_file(self, data_file, index_list, desc=""):
        X = list()
        y = list()
        for idx in tqdm(index_list, desc="Loading data from h5 file." + desc):
            X.append(data_file["X"][str(idx)])
            y.append(data_file["Y"][str(idx)])
        return {"X": X, "y": y}
