import numpy as np
from scipy import stats
from tree_ import DecisionTreeClassifier_, DecisionTreeRegressor_
from metrics_ import r2_score_


class RandomForestClassifier_:
    """
    RandomForestClassifier with bootstrap sample only
    """

    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=None,
                 min_samples_leaf=1,
                 min_impurity_decrease=0.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        self.trees = [
            DecisionTreeClassifier_(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_impurity_decrease=min_impurity_decrease
            ) for _ in range(n_estimators)]

    def fit(self, X, y):
        for tree in self.trees:
            idx = np.random.choice(len(X), len(X))
            X_sample = X[idx]
            y_sample = y[idx]
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        preds = []
        for tree in self.trees:
            pred = tree.predict(X)
            preds.append(pred)
        y_pred = stats.mode(preds)[0][0]
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y == y_pred).mean()


class RandomForestRegressor_:
    """
    RandomForestRegressor with bootstrap sample only
    """

    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=None,
                 min_samples_leaf=1,
                 min_impurity_decrease=0.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        self.trees = [
            DecisionTreeRegressor_(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_impurity_decrease=min_impurity_decrease
            ) for _ in range(n_estimators)]

    def fit(self, X, y):
        for tree in self.trees:
            idx = np.random.choice(len(X), len(X))
            X_sample = X[idx]
            y_sample = y[idx]
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        preds = []
        for tree in self.trees:
            pred = tree.predict(X)
            preds.append(pred)
        y_pred = np.array(preds).mean(axis=0)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score_(y, y_pred)
