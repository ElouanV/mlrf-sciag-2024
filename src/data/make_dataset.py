import numpy as np
import pickle
import os


class CIFARLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None

    def load_data(self):
        self._load_train_data()
        self._load_test_data()
        self._preprocess_data()

    def _load_batch(self, file_path):
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        return images, labels

    def _load_train_data(self):
        train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        images_list = []
        labels_list = []
        for file in train_files:
            file_path = os.path.join(self.data_dir, file)
            images, labels = self._load_batch(file_path)
            images_list.append(images)
            labels_list.append(labels)
        self.train_data = np.concatenate(images_list)
        self.train_labels = np.concatenate(labels_list)

    def _load_test_data(self):
        test_file = 'test_batch'
        file_path = os.path.join(self.data_dir, test_file)
        self.test_data, self.test_labels = self._load_batch(file_path)

    def _preprocess_data(self):
        # Perform preprocessing steps here
        pass
