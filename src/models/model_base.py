import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import os
import joblib
from utils import check_dir

class model_base:
    def __init__(self, save_path, model_name, verbose=True):
        self.save_path = os.path.join(save_path, model_name)
        check_dir(self.save_path)
        self.model = None
        self.model_name = model_name
        self.label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.verbose = verbose

    def fit(self, X, y):
        if self.verbose:
            print('Fitting model...')
        self.model.fit(X, y)
        if self.verbose:
            print('Model fitted.')
        self.save()

    def predict(self, X):
        return self.model.predict(X)

    def save(self):
        joblib.dump(self.model, self.save_path)

    def load(self):
        self.model = joblib.load(self.save_path)
        return self.model

    def fine_tune(self, grid, X, y, cv=5, n_jobs=-1, verbose=1):
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(self.model, grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        self.save()
        return self.model

    def test(self, X_test, Y_test):
        model_name = self.model_name
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(Y_test, y_pred)
        print(f'{model_name} Accuracy: {(accuracy * 100):.2f}%')
        conf_mat = confusion_matrix(Y_test, y_pred)

        y_pred_proba = self.model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(Y_test, y_pred_proba[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(16, 12))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
        plt.xticks(np.arange(10))
        plt.xticks(np.arange(10), self.label_name, rotation=45)
        plt.yticks(np.arange(10), self.label_name, rotation=0)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Flatten' + model_name + ' - ' + str('Accuracy %.2f' % (accuracy * 100)) + '%' + '\nConfusion Matrix')
        plt.savefig(f'figures/confusion_matrix_{model_name}_flatten_ft.png')
        plt.show()

        plt.figure(figsize=(16, 12))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.title('Flatten: ' + model_name + str(' - Accuracy %.2f' % (accuracy * 100)) + '%' + '\nROC Curve',
                  fontsize=20)
        plt.legend(loc="lower right")
        plt.savefig(f'figures/roc_{model_name}_flatten_ft.png')
        plt.show()
