import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score

import copy

from src.scripts.experimental.wrappers.experiment_model import BaseModel

class KFoldExperiment():
    def __init__(self, model: BaseModel, n_folds: int=5, n_repeats: int=2, seed: int=42):
        self.model = model
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.seed = seed
        
    def _calculate_metrics(self, y_pred, y_true):
        pass
        
    def run(self):
        # temporary, since the input is yet in an unknown format (images + text)
        X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        y = np.array([0, 0, 1, 1])
        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_folds, 
            n_repeats=self.n_repeats,
            random_state=self.seed
            )
        
        # n_folds * n_repeats - since there is only one model
        bac_scores = np.zeros((self.n_folds * self.n_repeats))
        
        for i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            clf = copy.deepcopy(self.model) # should temporary work
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            bac_scores[i] = balanced_accuracy_score(y_test, preds)
            # metrics = self._calculate_metrics(preds, y_test)
    