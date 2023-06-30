from src.data.make_dataset import CIFARLoader
import numpy as np

from src.features.build_features import ColorHist
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.dummy import DummyClassifier

# data_src = './data/cifar-10-batches-py/'
# # data_dir = './data/preprocessed'
# # batches = [1, 2, 3, 4, 5]
# # cifar_loader = CIFARLoader(data_src, data_dir, batches=batches)
# # train_data, train_labels = cifar_loader.get_train()
# # print(f'Train shape: {train_data.shape}')
# # test_data, test_labels = cifar_loader.get_test()
# # print(f'Test shape: {test_data.shape}')
# #
# # from src.utils import *
# #
# # build_sift_desc('./data/preprocessed')

from src.features.build_features import SIFTBoVW
cifar_labels_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                      'truck']
from sklearn.model_selection import train_test_split
sift_bovw = SIFTBoVW(num_clusters=512, data_dir='./data/preprocessed/sift')
print('Getting train and test descriptors ...')
X_train, y_train = sift_bovw.get_train()
print(f'Train desc shape: {X_train.shape}')
print(f'Train labels shape: {y_train.shape}')

def plot_results(model, X_train_color, Y_train_color, X_test_color, Y_test_color, labels_name):
    model_name = model.__class__.__name__

    Y_pred_color = model.predict(X_test_color)

    accuracy = accuracy_score(Y_test_color, Y_pred_color)
    conf_mat = confusion_matrix(Y_test_color, Y_pred_color)

    Y_pred_proba = model.predict_proba(X_test_color)
    fpr, tpr, _ = roc_curve(Y_test_color, Y_pred_proba[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Création de la figure et des sous-graphiques
    # Deuxième graphique : Heatmap de la matrice de confusion
    plt.figure(figsize=(16, 12))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    plt.xticks(np.arange(10))
    # Set labels for the x-axis ticks
    plt.xticks(np.arange(10), labels_name, rotation=45)
    plt.yticks(np.arange(10), labels_name, rotation=0)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('BoVW: ' + model_name + ' - ' + str('Accuracy %.2f' % (accuracy * 100)) + '%' + '\nConfusion Matrix')
    plt.savefig(f'figures/confusion_matrix_{model_name}_bovw_ft.png')
    plt.show()

    # Troisième graphique : Courbe ROC
    plt.figure(figsize=(16, 12))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('BoVW: ' + model_name + ' - ' + str('Accuracy %.2f' % (accuracy * 100)) + '%' + '\nROC Curve', fontsize=20)
    plt.legend(loc="lower right")
    plt.savefig(f'figures/roc_{model_name}_bovw_ft.png')
    plt.show()

import warnings
warnings.filterwarnings('ignore')
def train_models():
    models = {
        'dummy': DummyClassifier(strategy='most_frequent', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42)}
    for model_name, model in models.items():
        print(f'Training {model_name}')
        model.fit(X_train, y_train)
        plot_results(model, X_train, y_train, X_test, y_test, cifar_labels_names)


def fine_tune():
    model = LogisticRegression(
        random_state=42,
    )
    param_grid = {
        'C': [0.1, 1, 10,0.01],
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'max_iter': [ 5000]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
    plot_results(grid_search.best_estimator_, X_train, y_train, X_test, y_test, cifar_labels_names)

X_test, y_test = sift_bovw.get_test()
print(f'Test desc shape: {X_test.shape}')
print(f'Test labels shape: {X_test.shape}')

fine_tune()