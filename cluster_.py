import numpy as np
from metrics_ import silhouette_score_


class KMeans_:
    """
    支持KMeans算法
    """

    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_features_ = None
        self.labels_ = None
        self.cluster_centers_ = None

    def _get_label(self, X, centers):  # 根据给定centers计算到各标签距离，从而求解分类的标签
        dis_K = np.zeros([len(X), self.n_clusters])
        for i in range(self.n_clusters):
            dis_K[:, i] = np.linalg.norm(centers[i] - X, axis=1)
        return np.argmin(dis_K, axis=1)

    def _get_center(self, X, label):
        centers = np.zeros([self.n_clusters, self.n_features_])
        for i in range(self.n_clusters):
            centers[i] = X[label == i, :].mean(axis=0)
        return centers

    def fit(self, X):
        self.n_features_ = np.array(X).shape[1]
        centers = (X.max(axis=0) - X.min(axis=0)) * np.random.rand(self.n_clusters, self.n_features_) + X.min(axis=0)
        labels = None
        for _ in range(self.max_iter):
            labels = self._get_label(X, centers)
            centers = self._get_center(X, labels)
        self.labels_ = labels
        self.cluster_centers_ = centers
        return self

    def predict(self, X):
        return self._get_label(X, self.cluster_centers_)

    def score(self, X):
        y = self.predict(X)
        return silhouette_score_(X, y)
