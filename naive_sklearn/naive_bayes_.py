import numpy as np
from scipy.stats import norm
from collections import Counter
from metrics_ import *


class GaussianNB_:
    """
    高斯朴素贝叶斯
    """

    def __init__(self, priors=None, var_smoothing=1e-9):
        self.class_prior_ = priors
        self.var_smoothing = var_smoothing
        self.theta_ = None
        self.sigma_ = None
        self.class_count_ = None
        self.classes_ = None

    def fit(self, X_train, y_train):  # 训练函数
        self.classes_ = np.unique(y_train)
        counter = Counter(y_train)
        self.class_count_ = np.array([counter[cls] for cls in self.classes_])
        if not self.class_prior_:
            self.class_prior_ = self.class_count_ / len(y_train)
        self.theta_ = np.empty((len(self.classes_), X_train.shape[1]))
        self.sigma_ = np.empty((len(self.classes_), X_train.shape[1]))
        for cls in self.classes_:
            self.theta_[cls] = np.mean(X_train[y_train == cls, :], axis=0)  # 各标签下，各特征分布均值
            self.sigma_[cls] = np.std(X_train[y_train == cls, :], axis=0)  # 各标签下，各特征分布方差
        return self

    def predict_proba_one(self, x):
        probs = np.empty_like(self.classes_, dtype=float)
        for cls in self.classes_:
            probs[cls] = self.class_prior_[cls] * np.product(norm.pdf(x, loc=self.theta_[cls], scale=self.sigma_[cls]))
        return probs / np.sum(probs)

    def predict_proba(self, X):
        assert X.shape[1] == self.theta_.shape[1], 'feature numbers dis-match'
        proba_pre = np.empty((len(X), len(self.classes_)))
        for i, x in enumerate(X):
            proba_pre[i] = self.predict_proba_one(x)
        return proba_pre

    def predict(self, X):  # 预测函数
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X_test, y_test):  # 评分函数
        y_pre = self.predict(X_test)
        return accuracy_score_(y_pre, y_test)


class MultinomialNB_:
    """
    多项式朴素贝叶斯
    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.probs_ = None
        self.class_count_ = None
        self.classes_ = None
        self.feature_count_ = None

    def fit(self, X_train, y_train):
        self.classes_ = np.unique(y_train)
        counter = Counter(y_train)
        self.class_count_ = np.array([counter[cls] for cls in self.classes_])
        if not self.class_prior:
            if self.fit_prior:
                self.class_prior = self.class_count_ / len(y_train)
            else:
                self.class_prior = np.ones(len(self.classes_)) / len(self.classes_)
        self.feature_count_ = np.empty((len(self.classes_), X_train.shape[1]), dtype=int)
        self.probs_ = np.empty((len(self.classes_), X_train.shape[1]), dtype=dict)
        for fea_idx in range(X_train.shape[1]):
            features = np.unique(X_train[:, fea_idx])
            for cls in self.classes_:
                features_cls = X_train[y_train == cls, fea_idx]
                self.feature_count_[cls, fea_idx] = len(features_cls)
                counter = Counter(features_cls)
                self.probs_[cls][fea_idx] = {
                    feature: (self.alpha + counter[feature]) / (len(features_cls) + len(features) * self.alpha) for
                    feature in features}
        return self

    def predict_proba_one(self, x):
        probs = np.empty_like(self.classes_, dtype=float)
        default_prob = 1 / np.sum(self.class_count_)  # 以训练样本中 1/样本数量作为未出现的默认概率
        for cls in self.classes_:
            probs[cls] = self.class_prior[cls]
            for fea_idx, fea in enumerate(x):
                probs[cls] *= self.probs_[cls][fea_idx].get(fea, default_prob)
        return probs / np.sum(probs)

    def predict_proba(self, X):
        assert X.shape[1] == self.probs_.shape[1], 'feature numbers dis-match'
        proba_pre = np.empty((len(X), len(self.classes_)))
        for i, x in enumerate(X):
            proba_pre[i] = self.predict_proba_one(x)
        return proba_pre

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X_test, y_test):
        y_pre = self.predict(X_test)
        return accuracy_score_(y_pre, y_test)
