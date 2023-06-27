import os
import numpy as np
import pickle
import cv2

def checkdir(path):
    r"""
    Check if directory exists, if not, create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)


SIFTGEO_DTYPE = np.dtype([
    ("x", "<f4"),
    ("y", "<f4"),
    ("scale", "<f4"),
    ("angle", "<f4"),
    ("mi11", "<f4"),
    ("mi12", "<f4"),
    ("mi21", "<f4"),
    ("mi22", "<f4"),
    ("cornerness", "<f4"),
    ("desdim", "<i4"),
    ("component", "<u1", 128)
])


def siftgeo_read_full(path):
    return np.fromfile(path, dtype=SIFTGEO_DTYPE)
def siftgeo_read_label(path):
    filename = path.split('/')[-1]
    filename = filename.split('.')[0]
    label = filename.split('_')[-1]
    return int(label)

def siftgeo_read_desc(path):
    desc = siftgeo_read_full(path)["component"]
    if desc.size == 0:
        desc = np.zeros((0, 128), dtype='uint8')
    return desc


def build_sift_desc(data_dir):
    # For all images in data_dir/train and data_dir/test, extract SIFT descriptors and save them in data_dir/sift/train and data_dir/sift/test as .siftgeo files
    sift = cv2.SIFT_create()
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    sift_train_path = os.path.join(data_dir, 'sift', 'train')
    sift_test_path = os.path.join(data_dir, 'sift', 'test')
    checkdir(sift_train_path)
    checkdir(sift_test_path)
    for file in os.listdir(train_path):
        file_name = file.split('.')[0]
        with open(os.path.join(train_path, file), 'rb') as f:
            image = pickle.load(f)
        image = getImage(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, desc = sift.detectAndCompute(gray_image, None)
        if desc is not None:
            with open(os.path.join(sift_train_path, f'{file_name}.siftgeo'), 'wb') as f:
                desc.tofile(f)
    for file in os.listdir(test_path):
        file_name = file.split('.')[0]
        with open(os.path.join(test_path, file), 'rb') as f:
            image = pickle.load(f)
        image = getImage(image)
        image = cv2.resize(image, (32, 32))
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, desc = sift.detectAndCompute(gray_image, None)
        if desc is not None:
            with open(os.path.join(sift_test_path, f'{file_name}.siftgeo'), 'wb') as f:
                desc.tofile(f)


def getImage(img_mat, plot=False):
    '''
        @description returns a 32x32 image given a single row
        repr of the image
        _Optionally plots the image_
        @param img_mat -> np.array: |img_mat| = (3072, ) OR (3072, 1)
        @param plot -> bool: whether to plot it or not
        @return image_repr: np.ndarray |image_repr| = (32, 32, 3)
    '''
    assert img_mat.shape in [(3072,), (3072, 1)] # sanity check
    r_channel = img_mat[:1024].reshape(32, 32)
    g_channel = img_mat[1024: 2 * 1024].reshape(32, 32)
    b_channel = img_mat[2 * 1024:].reshape(32, 32)
    image_repr = np.stack([r_channel, g_channel, b_channel], axis=2)
    assert image_repr.shape == (32, 32, 3) # sanity check
    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(image_repr), plt.show(block=False)

    return image_repr

