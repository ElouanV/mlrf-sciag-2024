from sklearn.linear_model import LogisticRegression
from src.models.model_base import model_base

cifar_labels_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                      'truck']


class LogisticReg(model_base):
    def __init__(self, save_path, model_name='logistic_reg', penalty='l2', dual=False, tol=0.0001, C=1.0,
                 fit_intercept=True,
                 intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
                 max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None):
        super().__init__(save_path, model_name)
        self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                        intercept_scaling=intercept_scaling, class_weight=class_weight,
                                        random_state=random_state, solver=solver, max_iter=max_iter,
                                        multi_class=multi_class, verbose=verbose, warm_start=warm_start,
                                        n_jobs=n_jobs, l1_ratio=l1_ratio)
