import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin


class EDEL(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier: BaseEstimator, n_splits=3, random_state=42):
        self.base_classifier_template = base_classifier
        self.n_splits = n_splits
        self.random_state = random_state
        self.classifiers = []

    def train_ensemble_model(self, X, y):
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_index, val_index in kf.split(X, y):
            X_train_fold, y_train_fold = X.iloc[train_index], y.iloc[train_index]
            X_val_fold, y_val_fold = X.iloc[val_index], y.iloc[val_index]

            classifier = copy.deepcopy(self.base_classifier_template)
            classifier.fit(X_val_fold, y_val_fold)

            val_predictions = classifier.predict(X_train_fold)
            incorrect_indices = (val_predictions != y_train_fold)

            if any(incorrect_indices):
                X_incorrect = X_train_fold[incorrect_indices]
                y_incorrect = y_train_fold[incorrect_indices]
                updated_X_train = pd.concat([X_val_fold, X_incorrect])
                updated_y_train = np.concatenate([y_val_fold, y_incorrect])

                classifier_updated = copy.deepcopy(self.base_classifier_template)
                classifier_updated.fit(updated_X_train, updated_y_train)
                self.classifiers.append(classifier_updated)
            else:
                self.classifiers.append(classifier)

    def fit(self, X_train, y_train):
        self.train_ensemble_model(X_train, y_train)
        return self

    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_vote

    def predict_proba(self, X):
        probas = np.mean([clf.predict_proba(X) for clf in self.classifiers], axis=0)
        return probas
