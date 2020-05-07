import numpy as np
from metrics_ import *


def train_test_split_(X, y, train_size=0.8, test_size=0.2):
    assert len(X) == len(y) and X.ndim == 2 and y.ndim == 1, 'X or y with wrong dimensions'
    assert train_size + test_size == 1, 'train_size + test_size should be equals 1'
    n_samples = len(X)
    if train_size != 0.8:
        k = int(n_samples * train_size)
    else:
        k = int(n_samples * (1 - test_size))
    idxs = np.random.permutation(range(n_samples))
    X_train = X[idxs[:k]]
    X_test = X[idxs[k:]]
    y_train = y[idxs[:k]]
    y_test = y[idxs[k:]]
    return X_train, X_test, y_train, y_test


def cross_val_score_(estimator, X, y=None, cv=5):
    assert isinstance(cv, int) and cv > 1, 'cv should be a positive integer'
    scores = [0] * cv
    k = len(X) // cv
    est = estimator
    for i in range(cv):
        X_train = np.vstack([X[:i * k], X[(i + 1) * k:]])
        X_test = X[i * k:(i + 1) * k]
        y_train = np.concatenate([y[:i * k], y[(i + 1) * k:]])
        y_test = y[i * k:(i + 1) * k]
        est.fit(X_train, y_train)
        scores[i] = est.score(X_test, y_test)
    return scores


class GridSearchCV_:
    """
    根据给定参数列表，执行网格搜索返回最优参数
    """

    def __init__(self, estimator, param_grid, scoring=None, cv=None):
        self.estimator = estimator
        self.params = self._get_all_params(param_grid)
        self.scoring = scoring
        if cv is not None:
            self.cv = cv
        else:
            self.cv = 5
        self.best_param_ = None
        self.best_score = None
        self.scores = []

    def _get_all_params(self, param_grid):
        # 获取所有可能的参数组合
        if isinstance(param_grid, dict):
            param_grid = [param_grid]
        params = []
        for param in param_grid:
            keys = param.keys()
            vals = [[]]
            for key in keys:
                vals = [v + [p] for v in vals for p in param[key]]
            params.extend([{k: v for k, v in zip(keys, val)} for val in vals])
        return params

    def fit(self, X, y=None):
        # score_metrics = {'r2': r2_score_,
        #                  'mean_squared_error': mean_squared_error_
        #                  }
        # if self.scoring:
        #     score = score_metrics[self.scoring]
        # else:
        #     score = estimator.score()
        for param in self.params:
            score = cross_val_score_(self.estimator(**param), X, y, cv=self.cv)
            self.scores.append(np.mean(score))
        self.best_score = np.max(self.scores)
        self.best_param_ = self.params[np.argmax(self.scores)]
        return self

    def score(self, ):
        pass
