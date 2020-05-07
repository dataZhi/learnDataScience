import numpy as np
from itertools import combinations, combinations_with_replacement
from collections import Counter


class StandardScaler_:
    """对特征矩阵按 列 实现标准化处理"""

    def __init__(self):
        self.mu_ = None
        self.sigma_ = None

    def fit(self, X):
        self.mu_ = np.mean(X, axis=0)
        self.sigma_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        assert self.mu_ is not None and self.sigma_ is not None, 'You need fit before transform!'
        return (X - self.mu_) / self.sigma_

    def inverse_transform(self, X):
        assert self.mu_ is not None and self.sigma_ is not None, 'You need fit before transform!'
        return X * self.sigma_ + self.mu_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MinMaxScaler_:
    """对特征矩阵按 列 实现归一化处理"""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X_train):
        self.min_ = np.min(X_train, axis=0)
        self.max_ = np.max(X_train, axis=0)
        return self

    def transform(self, X):
        assert self.min_ is not None and self.max_ is not None, 'You need fit before transform!'
        return (X - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, X):
        assert self.min_ is not None and self.max_ is not None, 'You need fit before transform!'
        return X * (self.max_ - self.min_) + self.min_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class PolynomialFeatures_:
    """
    执行多项式处理特征
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        """
        :param degree: 阶数
        :param interaction_only:仅拟合交叉项
        :param include_bias: 包含全1列
        """
        assert isinstance(degree, int) and degree > 1, 'degree error!'
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.n_samples = None
        self.n_features = None

    def fit(self, X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        return self

    def transform(self, X):
        assert self.n_features and self.n_samples, 'fit before transform!'
        assert X.shape == (self.n_samples, self.n_features), 'X should has the right dimensions with fit data'
        XX = X.copy()
        if self.include_bias:
            XX = np.hstack([np.ones((self.n_samples, 1)), XX])
        comb = combinations if self.interaction_only else combinations_with_replacement
        for i in range(2, self.degree + 1):
            idxs = comb(range(X.shape[1]), r=i)
            for idx in idxs:
                tmp = np.ones(len(X))
                for ii in idx:
                    tmp *= X[:, ii]
                XX = np.hstack([XX, tmp.reshape(-1, 1)])
        return XX

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_names(self, input_features=None):
        assert self.n_features and self.n_samples, 'fit first'
        assert input_features is None or len(input_features) == self.n_features, 'wrong length of input_features'
        if not input_features:
            input_features = ['X'+str(i) for i in range(self.n_features)]
        feature_names = input_features[:]
        if self.include_bias:
            feature_names.insert(0, '1')
        comb = combinations if self.interaction_only else combinations_with_replacement
        for i in range(2, self.degree + 1):
            idxs = comb(range(self.n_features), r=i)  # 所有r阶的索引组合
            for idx in idxs:
                cnter = Counter(idx)
                name = ['%s^%d' % (input_features[ii], cnt) if cnt > 1 else input_features[ii] for ii, cnt in cnter.items()]
                feature_names.append(' '.join(name))
        return feature_names


class LabelEncoder_:
    """
    标签编码，仅限于单列
    """

    def __init__(self):
        self.classes_ = None
        self.maps_ = None

    def fit(self, y):
        self.classes_ = np.array(np.sort(np.unique(y)), dtype=object)
        self.maps_ = {self.classes_[i]: i for i in range(len(self.classes_))}
        return self

    def transform(self, y):
        assert all(yy in self.maps_ for yy in np.unique(y)), 'y contains previously unseen labels'
        return np.array([self.maps_[yy] for yy in y])

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        assert np.max(y) < len(self.classes_), 'y is longer than ever before labels'
        return np.array([self.classes_[yy] for yy in y])


class OrdinalEncoder_:
    """
    仅限于二维矩阵的数值编码
    """

    def __init__(self):
        self.n_features = None
        self.categories_ = None
        self.maps_ = None

    def fit(self, X):
        X = np.array(X)
        self.n_features = X.shape[1]
        self.categories_ = []
        self.maps_ = []
        for i in range(self.n_features):
            le = LabelEncoder_().fit(X[:, i])
            self.categories_.append(le.classes_)
            self.maps_.append(le.maps_)
        return self

    def transform(self, X):
        X = np.array(X)
        assert self.n_features == X.shape[1], 'number of features error'
        XX = np.empty(X.shape)
        for i in range(self.n_features):
            if any(yy not in self.maps_[i] for yy in np.unique(X[:, i])):
                raise ValueError('found previously unseen feature')
            XX[:, i] = np.array([self.maps_[i][yy] for yy in X[:, i]])
        return XX

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.array(X)
        XX = np.empty(X.shape)
        for i in range(self.n_features):
            if np.max(X[:, i]) > len(self.categories_[i]):
                raise ValueError('found previously unseen feature')
            XX[:, i] = np.array([self.categories_[i][j] for j in X[:, i]])
        return XX


class OneHotEncoder_:
    """
    独热编码
    """

    def __init__(self):
        self.n_features = None
        self.categories_ = None
        self.maps_ = None

    def fit(self, X):
        self.n_features = np.array(X).shape[1]
        self.categories_ = []
        self.maps_ = []
        for i in range(self.n_features):
            le = LabelEncoder_().fit(np.array(X)[:, i])
            self.categories_.append(le.classes_)
            self.maps_.append(le.maps_)
        return self

    def transform(self, X):
        assert np.array(X).shape[1] == self.n_features, 'dimension error'
        XX = np.empty((len(X), 0))
        for i in range(self.n_features):
            for c in self.categories_[i]:
                XX = np.hstack([XX, np.array(np.array(X)[:, i] == c, dtype=int).reshape(-1, 1)])
        return XX

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        assert np.array(X).shape[1] == np.sum([len(c) for c in self.categories_]), 'dimension error'
        XX = np.empty((len(X), self.n_features), dtype=object)
        start = 0
        for col, cate in enumerate(self.categories_):
            idx = np.array(X)[:, start:start+len(cate)].dot(np.arange(len(cate)))
            XX[:, col] = np.array([cate[i] for i in np.array(idx, dtype=int)])
            start += len(cate)
        return XX
