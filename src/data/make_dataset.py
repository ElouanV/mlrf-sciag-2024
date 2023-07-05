import numpy as np
import pickle
import os
from src.utils import *
import shutil

def _clean_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=False, onerror=None)


class CIFARLoader:
    def __init__(self, data_src, data_dir, batches=[1,2,3,4,5]):
        self.data_dir = data_dir
        self.data_src = data_src
        self.batches = batches
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.val_data = None
        self.val_labels = None
        _clean_folder(self.data_dir)
        self._load_data()
        self._save_data()

    def _load_data(self):
        data = []
        labels = []
        for batch in self.batches:
            with open(os.path.join(self.data_src, f'data_batch_{batch}'), 'rb') as f:
                batch_data = pickle.load(f, encoding='bytes')
            data.append(batch_data[b'data'])
            labels.append(batch_data[b'labels'])
        self.train_data = np.vstack(data)
        self.train_labels = np.hstack(labels)
        with open(os.path.join(self.data_src, 'test_batch'), 'rb') as f:
            batch_data = pickle.load(f, encoding='bytes')
        data = batch_data[b'data']
        labels = batch_data[b'labels']
        self.test_data = np.vstack(data)
        self.test_labels = np.hstack(labels)


    def _save_data(self):
        # SAve train data in data_dir/train
        train_path = os.path.join(self.data_dir, 'train')
        checkdir(train_path)
        for i, (data, label) in enumerate(zip(self.train_data, self.train_labels)):
            with open(os.path.join(train_path, f'{i}_{label}.pkl'), 'wb') as f:
                pickle.dump(data, f)
        # Save test data in data_dir/test
        test_path = os.path.join(self.data_dir, 'test')
        checkdir(test_path)
        for i, (data, label) in enumerate(zip(self.test_data, self.test_labels)):
            with open(os.path.join(test_path, f'{i}_{label}.pkl'), 'wb') as f:
                pickle.dump(data, f)

    def get_train(self):
        train_path = os.path.join(self.data_dir, 'train')
        train_data = []
        train_labels = []
        for file in os.listdir(train_path):
            with open(os.path.join(train_path, file), 'rb') as f:
                data = pickle.load(f)
            train_data.append(data)
            train_labels.append(file.split('_')[1].split('.')[0])
        return np.array(train_data), np.array(train_labels)

    def get_test(self):
        test_path = os.path.join(self.data_dir, 'test')
        test_data = []
        test_labels = []
        for file in os.listdir(test_path):
            with open(os.path.join(test_path, file), 'rb') as f:
                data = pickle.load(f)
            test_data.append(data)
            test_labels.append(file.split('_')[1].split('.')[0])
        return np.array(test_data), np.array(test_labels)
