import numpy as np

from src.features.build_features import Flatten
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
# logistic regression
from sklearn.linear_model import LogisticRegression
import seaborn as sns
#dummy classifier
from sklearn.dummy import DummyClassifier


flatten = Flatten(data_dir='./data/preprocessed')
cifar_labels_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                      'truck']
X_train, y_train = flatten.get_train()
X_test, y_test = flatten.get_test()
print(f'Train desc shape: {X_train.shape}')
print(f'Train labels shape: {y_train.shape}')
print(f'Test desc shape: {X_test.shape}')
print(f'Test labels shape: {y_test.shape}')


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
    plt.title('Flatten' + model_name + ' - ' + str('Accuracy %.2f' % (accuracy * 100)) + '%' + '\nConfusion Matrix')
    plt.savefig(f'figures/confusion_matrix_{model_name}_flatten.png')
    plt.show()

    # Troisième graphique : Courbe ROC
    plt.figure(figsize=(16, 12))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('Flatten: ' + model_name + str(' - Accuracy %.2f' % (accuracy * 100)) + '%' + '\nROC Curve', fontsize=20)
    plt.legend(loc="lower right")
    plt.savefig(f'figures/roc_{model_name}_flatten.png')
    plt.show()

def train_models():
    models = {
        'dummy': DummyClassifier(strategy='most_frequent', random_state=42),
        'sgd': SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None),
        'SVC': SVC(kernel='linear', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 100), random_state=42),
        'logistic_regression': LogisticRegression(random_state=42)}
    for model_name, model in models.items():
        print(f'Training {model_name}')
        model.fit(X_train, y_train)
        plot_results(model, X_train, y_train, X_test, y_test, cifar_labels_names)


def fine_tune_model():
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 100), random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42)}
    grids = {'LinearSVC': {'C': [0.1, 1, 0.01, 0.001]},
             'random_forest': {'n_estimators': [10, 50, 100, 200, 500], 'max_depth': [5, 10, 20, 50]},
             'MLP': {'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)]},
             'AdaBoost': {'n_estimators': [10, 50, 100, 200, 500]},
             'logistic_regression': {'C': [0.1, 1, 0.01, 0.001]}}
    best_models = {}
    best_scores = {}
    for model_name, model in models.items():
        print(f'Fine tuning {model_name}')
        grid = grids[model_name]
        grid_clf = GridSearchCV(model, grid, cv=5, n_jobs=-1)
        grid_clf.fit(X_train, y_train)
        print(f'Best params: {grid_clf.best_params_}')
        print(f'Best score: {grid_clf.best_score_}')
        models[model_name] = grid_clf.best_estimator_
        best_models[model_name] = grid_clf.best_estimator_
        best_scores[model_name] = grid_clf.best_score_
        plot_results(grid_clf.best_estimator_, X_train, y_train, X_test, y_test, cifar_labels_names)
    return best_models, best_scores


train_models()