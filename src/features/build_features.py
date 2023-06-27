
from sklearn.cluster import KMeans
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class SIFTBoVW:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.kmeans = None
        self.scaler = None

    def fit(self, data):
        descriptors = self._extract_descriptors(data)
        self._cluster_descriptors(descriptors)

    def transform(self, data):
        descriptors = self._extract_descriptors(data)
        features = self._compute_features(descriptors)
        return features

    def _extract_descriptors(self, data):
        sift = cv2.xfeatures2d.SIFT_create()
        descriptors = []
        for image in data:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, desc = sift.detectAndCompute(gray_image, None)
            if desc is not None:
                descriptors.extend(desc)
        return np.array(descriptors)

    def _cluster_descriptors(self, descriptors):
        self.kmeans = KMeans(n_clusters=self.num_clusters)
        self.kmeans.fit(descriptors)

    def _compute_features(self, descriptors):
        features = np.zeros((len(descriptors), self.num_clusters), dtype=np.float32)
        for i, desc in enumerate(descriptors):
            labels = self.kmeans.predict(desc.reshape(1, -1))
            features[i, labels] += 1
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        return features





class HOGFeatureExtractor:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def extract_features(self, images):
        hog_features = []

        for image in images:
            # Convert the image to grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Calculate gradients using Sobel filter
            gradient_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0)
            gradient_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1)

            # Calculate gradient magnitude and orientation
            magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)

            # Create a histogram of oriented gradients
            histogram = np.zeros((self.cells_per_block[1], self.cells_per_block[0], self.orientations))
            cell_size_x = self.pixels_per_cell[0]
            cell_size_y = self.pixels_per_cell[1]
            angle_unit = 180 / self.orientations
            bin_count = int(360 / angle_unit)

            for i in range(self.cells_per_block[1]):
                for j in range(self.cells_per_block[0]):
                    cell_magnitude = magnitude[i * cell_size_y:(i + 1) * cell_size_y,
                                     j * cell_size_x:(j + 1) * cell_size_x]
                    cell_angle = angle[i * cell_size_y:(i + 1) * cell_size_y,
                                 j * cell_size_x:(j + 1) * cell_size_x]
                    histogram[i, j] = self.calculate_histogram(cell_magnitude, cell_angle, bin_count)

            # Flatten the histogram to obtain the feature vector
            features = histogram.ravel()
            hog_features.append(features)

        hog_features = np.array(hog_features)
        return hog_features

    def calculate_histogram(self, cell_magnitude, cell_angle, bin_count):
        histogram = np.zeros((bin_count,))
        angle_unit = 360 / bin_count

        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                magnitude = cell_magnitude[i, j]
                angle = cell_angle[i, j]
                bin_index = int(angle / angle_unit)
                histogram[bin_index] += magnitude

        return histogram


    def normalize_features(self, features):
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        return normalized_features

    def reduce_dimensionality(self, features, n_components):
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features)
        return reduced_features
