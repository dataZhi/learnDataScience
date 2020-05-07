import pandas as pd
import numpy as np
from metrics_ import *


class KNeighborsClassifier_:
    """
    K近邻分类，
    """

    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        """
        n_neighbors：近邻个数，默认为5个
        weights: 距离衡量指标
        p：距离范数指标，默认为2范数，即欧式距离
        """
        assert n_neighbors > 0 and isinstance(n_neighbors, int), \
            'n_neighbors must be a positive integer, "{}" is got'.format(n_neighbors)
        assert weights in ['uniform', 'distance'], 'weights optional is "uniform" or "distance"'
        assert p > 0 and isinstance(p, int), 'p must be a positive integer, "{}" is got'.format(p)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self._X = None
        self._y = None
        self.classes_ = None

    def fit(self, X, y):
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0], \
            'X and y must have the right dimensions'
        self._X = X
        self._y = y
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        assert X.shape[1] == self._X.shape[1], 'X should has the same number of features'
        proba = np.empty(shape=(len(X), len(self.classes_)), dtype=float)
        votes_zero = np.c_[self.classes_, np.zeros_like(self.classes_)]  # 避免某个类在K近邻中未出现，设置0默认0加权系数
        for i, x in enumerate(X):
            dists = np.linalg.norm(self._X - x, ord=self.p, axis=1)
            idx_k = np.argsort(dists)[:self.n_neighbors]  # 距离最近的K个索引信息
            votes = 1 / dists[idx_k] if self.weights == 'distance' else np.ones(self.n_neighbors)
            # 如果是按距离加权，则加权系数为其距离倒数；否则均为1
            y_k = self._y[idx_k]
            votes_kn = np.c_[y_k, votes]
            df = pd.DataFrame(np.vstack([votes_kn, votes_zero]), columns=['labels', 'votes'])
            proba[i] = np.array(df.groupby('labels').sum()).reshape(-1)  # groupby会自动对分类标签排序
        proba /= np.sum(proba, axis=1).reshape(-1, 1)
        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        y_pre = self.predict(X)  # 预测分类结果
        return accuracy_score_(y, y_pre)  # 返回正确率


class KNeighborsRegressor_:
    """
    K近邻回归
    """

    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        """
        n_neighbors：近邻个数，默认为5个
        weights: 距离衡量指标
        p：距离范数指标，默认为2范数，即欧式距离
        """
        assert n_neighbors > 0 and isinstance(n_neighbors, int), \
            'n_neighbors must be a positive integer, "{}" is got'.format(n_neighbors)
        assert weights in ['uniform', 'distance'], 'weights optional is "uniform" or "distance"'
        assert p > 0 and isinstance(p, int), 'p must be a positive integer, "{}" is got'.format(p)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self._X = None
        self._y = None

    def fit(self, X, y):
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0], \
            'X and y must have the right dimensions'
        self._X = X
        self._y = y
        return self

    def predict(self, X):
        y_pre = []
        for x in X:
            dists = np.linalg.norm( self._X - x, ord=self.p, axis=1)
            idx_k = np.argsort(dists)[:self.n_neighbors]  # 距离最近的K个索引信息
            votes_k = 1 / dists[idx_k] if self.weights == 'distance' else np.ones(self.n_neighbors)
            # 如果是按距离加权，则加权系数为其距离倒数；否则均为1
            y_k = self._y[idx_k]
            y_pre.append(np.sum(y_k * votes_k) / np.sum(votes_k))
        return np.array(y_pre)

    def score(self, X, y):
        y_pre = self.predict(X)  # 预测回归结果
        return r2_score_(y, y_pre)  # 返回R2评分
