from sklearn.cluster import KMeans
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
import os
from src.utils import *
import pickle
from tqdm import tqdm


class SIFTBoVW:
    def __init__(self, num_clusters=512, data_dir='./data/preprocessed/sift', verbose=False):
        self.num_clusters = num_clusters
        self.verbose = verbose
        self.kmeans = None
        self.scaler = None
        self.sift_desc_train = []
        self.sift_desc_test = []
        self.train_labels = []
        self.test_labels = []
        self.data_dir = data_dir
        for path in tqdm(os.listdir(os.path.join(data_dir, 'train'))):
            path = os.path.join(data_dir, 'train', path)
            self.sift_desc_train.append(siftgeo_read_desc(path))
            self.train_labels.append(siftgeo_read_label(path))
        self.sift_desc_train = np.vstack(self.sift_desc_train)
        print(f'Sift descriptors shape: {self.sift_desc_train.shape}')
        for path in tqdm(os.listdir(os.path.join(data_dir, 'test'))):
            path = os.path.join(data_dir, 'test', path)
            self.sift_desc_test.append(siftgeo_read_desc(path))
            self.test_labels.append(siftgeo_read_label(path))
        self.sift_desc_test = np.vstack(self.sift_desc_test)
        self.sift_desc_train = self.sift_desc_train.astype(np.double)
        self.sift_desc_test = self.sift_desc_test.astype(np.double)
        self.train_labels = np.array(self.train_labels)
        self.test_labels = np.array(self.test_labels)
        print(f'Loaded {len(self.sift_desc_train)} train descriptors')
        print(f'Loaded {len(self.sift_desc_test)} test descriptors')
        print(f'Fiting scaler ...')
        self.scaler = self._train_scaler()
        print('Done')
        # Aplly the scaler to the train and test descriptors
        self.sift_desc_train = self.scaler.transform(self.sift_desc_train)
        self.sift_desc_test = self.scaler.transform(self.sift_desc_test)
        print(f'Fiting PCA ...')
        self.pca = self._train_pca()
        print('Done')
        self.sift_desc_train = self.pca.transform(self.sift_desc_train)
        self.sift_desc_test = self.pca.transform(self.sift_desc_test)
        print(f'Shape after PCA: {self.sift_desc_train.shape}')
        print(f'Fiting KMeans ...')
        self.kmeans = self._train_kmeans()
        print('Done')
        self.normalizer = Normalizer(norm='l2', copy=True)

    def _train_pca(self):
        self.pca = PCA(n_components=0.95, random_state=42)
        self.pca.fit(self.sift_desc_train)
        return self.pca

    def _train_scaler(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.sift_desc_train)
        return self.scaler

    def _train_kmeans(self):
        # Train kmeans only on a portion of train_data
        print(self.sift_desc_train.shape)
        data = self.sift_desc_train[:1000]
        data = data.astype(np.double)
        print(data.shape)
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, verbose=self.verbose, n_init=1)
        self.kmeans.fit(data)
        return self.kmeans

    def _compute_train_descriptor(self, path):
        sifgeo_list_files = os.listdir(path)
        image_descriptor = np.zeros((len(sifgeo_list_files), self.kmeans.n_clusters), dtype=np.float32)
        for ii, descfile in tqdm(enumerate(sifgeo_list_files)):
            desc = siftgeo_read_desc(os.path.join(path, descfile))
            if desc is None or desc.shape[0] == 0:
                continue
            desc = self.scaler.transform(desc)
            desc = self.pca.transform(desc)
            desc = desc.astype(np.double)
            clabels = self.kmeans.predict(desc.astype('double'))
            descr_hist = np.histogram(clabels, bins=self.num_clusters, range=(0, self.num_clusters))[0]
            descr_hist = descr_hist / np.linalg.norm(descr_hist)
            descr_hist = np.sqrt(descr_hist)
            descr_hist = self.normalizer.transform(descr_hist.reshape(1, -1))
            image_descriptor[ii, :] = descr_hist
        return image_descriptor

    def _compute_test_descriptor(self, path):
        sifgeo_list_files = os.listdir(path)
        image_descriptor = np.zeros((len(sifgeo_list_files), self.kmeans.n_clusters), dtype=np.float32)
        for ii, descfile in tqdm(enumerate(sifgeo_list_files)):
            desc = siftgeo_read_desc(os.path.join(path, descfile))
            if desc is None or desc.shape[0] == 0:
                continue
            desc = self.scaler.transform(desc)
            desc = self.pca.transform(desc)
            clabels = self.kmeans.predict(desc)
            descr_hist = np.histogram(clabels, bins=self.num_clusters, range=(0, self.num_clusters))[0]
            descr_hist = descr_hist / np.linalg.norm(descr_hist)
            descr_hist = np.sqrt(descr_hist)
            descr_hist = self.normalizer.transform(descr_hist.reshape(1, -1))
            image_descriptor[ii, :] = descr_hist
        return image_descriptor

    def get_train(self):
        path = os.path.join(self.data_dir, 'train')
        X = self._compute_train_descriptor(path)
        return X, self.train_labels

    def get_test(self):
        path = os.path.join(self.data_dir, 'test')
        X = self._compute_test_descriptor(path)
        return X, self.test_labels


def color_hist(path):
    with open(path, 'rb') as f:
        img = pickle.load(f)
    img = getImage(img)
    # Turn img ndarray to cv2 image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / np.linalg.norm(hist)
    hist = np.sqrt(hist)

    return hist

def color_hist_label(path):
    filename = path.split('/')[-1]
    filename = filename.split('.')[0]
    label = filename.split('_')[-1]
    return int(label)

class ColorHist:
    def __init__(self, data_dir='./data/preprocessed/', verbose=False):
        self.data_dir = data_dir
        self.verbose = verbose
        self.train_labels = []
        self.test_labels = []
        self.train_hist = []
        self.test_hist = []
        for path in tqdm(os.listdir(os.path.join(data_dir, 'train'))):
            path = os.path.join(data_dir, 'train', path)
            self.train_hist.append(color_hist(path))
            self.train_labels.append(color_hist_label(path))
        for path in tqdm(os.listdir(os.path.join(data_dir, 'test'))):
            path = os.path.join(data_dir, 'test', path)
            self.test_hist.append(color_hist(path))
            self.test_labels.append(color_hist_label(path))
        self.train_hist = np.array(self.train_hist)
        self.test_hist = np.array(self.test_hist)
        self.train_labels = np.array(self.train_labels)
        self.test_labels = np.array(self.test_labels)

    def get_train(self):
        return self.train_hist, self.train_labels

    def get_test(self):
        return self.test_hist, self.test_labels


def flatten(path):
    with open(path, 'rb') as f:
        img = pickle.load(f)
    img = getImage(img)
    # Turn img ndarray to cv2 image to grayscale
    img = img.flatten()
    img = img / 255
    img = np.sqrt(img)
    return img

def flatten_label(path):
    filename = path.split('/')[-1]
    filename = filename.split('.')[0]
    label = filename.split('_')[-1]
    return int(label)
class Flatten:
    def __init__(self, data_dir='./data/preprocessed/', verbose=False):
        self.data_dir = data_dir
        self.verbose = verbose
        self.train_labels = []
        self.test_labels = []
        self.train_hist = []
        self.test_hist = []
        for path in tqdm(os.listdir(os.path.join(data_dir, 'train'))):
            path = os.path.join(data_dir, 'train', path)
            self.train_hist.append(flatten(path))
            self.train_labels.append(color_hist_label(path))
        for path in tqdm(os.listdir(os.path.join(data_dir, 'test'))):
            path = os.path.join(data_dir, 'test', path)
            self.test_hist.append(flatten(path))
            self.test_labels.append(color_hist_label(path))
        self.train_hist = np.array(self.train_hist)
        self.test_hist = np.array(self.test_hist)
        self.train_labels = np.array(self.train_labels)
        self.test_labels = np.array(self.test_labels)

    def get_train(self):
        return self.train_hist, self.train_labels

    def get_test(self):
        return self.test_hist, self.test_labels