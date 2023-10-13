import numpy as np


class PCA_:
    """
    根据协方差矩阵求解特征值，返回其给定特征值和特征向量
    基本思路：
        0. 对给定X(m, n)特征矩阵，通过某种变换，实现降维，即转换成 Y(m, k)而不至于损失过多信息，即寻找P(k, n)，实现 Y = X * P.T
        1. 对X的每一列特征去均值化，X = X - np.mean(X, axis=0)
        2. 记 cov(X) = X.T * X，即特征与特征之间的方差或协方差构成的矩阵，这是一个实对称矩阵
        3. cov(Y) = Y.T * Y = P * X.T * X * P.T = P * cov(X) * P.T，为使转换后的特征矩阵Y尽可能减低相关性，
           目标是使 cov(Y) 成为一个对角矩阵，所以根据 cov(X) 实对称矩阵的性质，P.T恰好为其特征向量的组成的矩阵，
           其转置即为所要寻找的线性变换矩阵 P
        4. 进一步地，cov(X) 的特征值即为解释方差权重，对其进行归一化，则可以得到前K个解释方差的比例，即为主成分的比重
    """

    def __init__(self, n_components=None):
        assert n_components is None or 0 < n_components < 1 or (isinstance(n_components, int) and n_components >= 1), \
            'n_components must be a positive integer or a fraction'
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None
        self.n_features_ = None
        self.n_samples_ = None
        self.mean_ = None

    def fit(self, X):
        self.n_samples_, self.n_features_ = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_demean = X - self.mean_
        eig, vec = np.linalg.eig(X_demean.T.dot(X_demean))
        explained_variance_ratio = eig / np.sum(eig)
        if self.n_components is None:
            self.n_components_ = min(X.shape)
        elif 0 < self.n_components < 1:
            K = 1
            while np.sum(explained_variance_ratio[:K]) < self.n_components:
                K += 1
            self.n_components_ = K
        else:
            self.n_components_ = self.n_components
        self.explained_variance_ = eig[:self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components_]
        self.components_ = vec.T[:self.n_components_]
        return self

    def transform(self, X):
        assert X.shape[1] == self.n_features_, 'X has the wrong dimensions'
        X_demean = X - self.mean_
        return X_demean.dot(self.components_.T)

    def inverse_transform(self, X):
        assert X.shape[1] == self.n_components_, 'X has the wrong dimensions'
        return X.dot(self.components_) + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
