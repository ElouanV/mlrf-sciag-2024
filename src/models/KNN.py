from sklearn.neighbors import KNeighborsClassifier
from src.models.model_base import model_base

cifar_labels_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                      'truck']


class KNN(model_base):
    def __init__(self, save_path, model_name='KNN', n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None, n_jobs=None):
        super().__init__(save_path, model_name)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                          leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params,
                                          n_jobs=n_jobs)
