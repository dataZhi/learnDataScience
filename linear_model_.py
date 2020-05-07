import numpy as np
from metrics_ import *
from preprocessing_ import StandardScaler_


class LinearRegression_:
    """
    支持最小二乘法、梯度下降法和随机梯度下降法3种求解，其中梯度下降法求解梯度的方式又包括依据数学推导公式计算和依据导数公式计算
    """

    def __init__(self, copy_X=True, normalize=True, solver='lsqr', solve_dJ='math', n_iters=1000, learning_rate=0.1,
                 tol=1e-3):
        # n_iters、 learning_rate、 tol在选用梯度下降和随机梯度下降法时有效， solve_dJ仅在选择梯度下降法时有效
        assert solver in ['lsqr', 'gd', 'sgd'], \
            'solver must be one of "lsqr", "gd", "sgd", and "{}" got!'.format(solver)
        assert solve_dJ in ['math', 'debug'], \
            'solve_dJ must be one of "math", "debug", and "{}" got!'.format(solve_dJ)
        self.copy_X = copy_X
        self.normalize = normalize
        self.solver = solver
        self.solve_dJ = solve_dJ
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X, y):
        if self.copy_X:
            X = X.copy()
        if self.normalize:
            X = StandardScaler_().fit_transform(X)
        X_b = np.c_[np.ones((len(X), 1)), X]  # 拼接构造含有常数项b的新样本数据
        if self.solver == 'lsqr':
            try:
                self._fit_lsq(X_b, y)
            except:
                self._fit_gd(X_b, y)
        elif self.solver == 'gd':
            self._fit_gd(X_b, y)
        else:
            self._fit_sgd(X_b, y)
        return self

    def _fit_lsq(self, X_b, y):  # 最小二乘法训练模型
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T.dot(y))  # (X^T * X)^-1 * (X^T * y)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

    def _fit_gd(self, X_b, y):  # 梯度下降法训练模型
        theta = np.ones(X_b.shape[1])
        if self.solve_dJ == 'math':  # 选择梯度求解方法
            solve_dJ = _dJ_gd_math
        else:
            solve_dJ = _dJ_gd_debug
        for _ in range(self.n_iters):
            dJ = solve_dJ(theta, X_b, y)
            theta -= self.learning_rate * dJ
            if np.linalg.norm(dJ) <= self.tol:  # 提前终止条件
                break
        self._theta = theta
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

    def _fit_sgd(self, X_b, y):  # 随机梯度下降法训练模型
        theta = np.zeros(X_b.shape[1])
        max_n_iters = max(self.n_iters // len(y), 10)  # 至少循环10次
        for i in range(max_n_iters):
            idxs = np.random.permutation(len(y))
            for idx in idxs:
                learning_rate = 1 / (i * len(y) + idx + 100)
                dJ = _dJ_sgd(theta, X_b, y, J='LR') # (X_b[idx].dot(theta) - y[idx]) * X_b[idx]  # 仅利用第idx个样本求解梯度
                theta -= learning_rate * dJ
                if np.linalg.norm(dJ) <= self.tol:  # 提前终止条件
                    break
        self._theta = theta
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

    def predict(self, X_test):
        if self.copy_X:
            X_test = X_test.copy()
        if self.normalize:
            X_test = StandardScaler_().fit_transform(X_test)
        return np.c_[np.ones((len(X_test), 1)), X_test].dot(self._theta)

    def score(self, X_test, y_test, scoring=None):
        # score采取R2评价指标
        score_metrics = {'r2': r2_score_,
                         'mean_squared_error': mean_squared_error_
                         }
        y_pre = self.predict(X_test)
        if scoring:
            return score_metrics[scoring](y_test, y_pre)
        return r2_score_(y_test, y_pre)


class Ridge_:
    """
    岭回归
    支持最小二乘法、梯度下降法和随机梯度下降法3种求解，其中梯度下降法求解梯度的方式又包括依据数学推导公式计算和依据导数公式计算
    """

    def __init__(self, alpha=1, copy_X=True, normalize=True,
                 solver='lsqr', solve_dJ='math', n_iters=1000, learning_rate=0.1, tol=1e-3):
        # n_iters、 learning_rate、 tol在选用梯度下降和随机梯度下降法时有效， solve_dJ仅在选择梯度下降法时有效
        assert solver in ['lsqr', 'gd', 'sgd'], \
            'solver must be one of "lsqr", "gd", "sgd", and "{}" got!'.format(solver)
        assert solve_dJ in ['math', 'debug'], \
            'solve_dJ must be one of "math", "debug", and "{}" got!'.format(solve_dJ)
        self.alpha = alpha
        self.copy_X = copy_X
        self.normalize = normalize
        self.solver = solver
        self.solve_dJ = solve_dJ
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X, y):
        if self.copy_X:
            X = X.copy()
        if self.normalize:
            X = StandardScaler_().fit_transform(X)
        X_b = np.c_[np.ones((len(X), 1)), X]  # 拼接构造含有常数项b的新样本数据
        if self.solver == 'lsqr':
            try:
                self._fit_lsq(X_b, y)
            except:
                self._fit_gd(X_b, y)
        elif self.solver == 'gd':
            self._fit_gd(X_b, y)
        else:
            self._fit_sgd(X_b, y)
        return self

    def _fit_lsq(self, X_b, y):  # 最小二乘法训练模型
        XT_X = X_b.T.dot(X_b) + self.alpha * np.eye(X_b.shape[1])
        self._theta = np.linalg.inv(XT_X).dot(X_b.T.dot(y))  # (X^T * X + alpha*I)^-1 * (X^T * y)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

    def _fit_gd(self, X_b, y):  # 梯度下降法训练模型
        theta = np.ones(X_b.shape[1])
        if self.solve_dJ == 'math':  # 选择梯度求解方法
            solve_dJ = _dJ_gd_math
        else:
            solve_dJ = _dJ_gd_debug
        for _ in range(self.n_iters):
            dJ = solve_dJ(theta, X_b, y, J='Ridge', alpha=self.alpha)
            theta -= self.learning_rate * dJ
            if np.linalg.norm(dJ) <= self.tol:  # 提前终止条件
                break
        self._theta = theta
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

    def _fit_sgd(self, X_b, y):  # 随机梯度下降法训练模型
        theta = np.zeros(X_b.shape[1])
        max_n_iters = max(self.n_iters // len(y), 10)  # 至少循环10次
        for i in range(max_n_iters):
            idxs = np.random.permutation(len(y))
            for idx in idxs:
                learning_rate = 1 / (i * len(y) + idx + 100)
                dJ = _dJ_sgd(theta, X_b, y, idx=idx, J='Ridge') # (X_b[idx].dot(theta) - y[idx]) * X_b[idx] + alpha * theta  # 仅利用第idx个样本求解梯度
                theta -= learning_rate * dJ
                if np.linalg.norm(dJ) <= self.tol:  # 提前终止条件
                    break
        self._theta = theta
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

    def predict(self, X_test):
        if self.copy_X:
            X_test = X_test.copy()
        if self.normalize:
            X_test = StandardScaler_().fit_transform(X_test)
        return np.c_[np.ones((len(X_test), 1)), X_test].dot(self._theta)

    def score(self, X_test, y_test, scoring=None):
        # score采取R2评价指标
        score_metrics = {'r2': r2_score_,
                         'mean_squared_error': mean_squared_error_
                         }
        y_pre = self.predict(X_test)
        if scoring:
            return score_metrics[scoring](y_test, y_pre)
        return r2_score_(y_test, y_pre)


class Lasso_:
    """
    lasso回归
    仅支持梯度下降法和随机梯度下降法3种求解
    """

    def __init__(self, alpha=1, copy_X=True, normalize=True,
                 solver='gd', solve_dJ='debug', n_iters=1000, learning_rate=0.1, tol=1e-3):
        # n_iters、 learning_rate、 tol在选用梯度下降和随机梯度下降法时有效， solve_dJ仅在选择梯度下降法时有效
        assert solver in ['gd', 'sgd'], \
            'solver must be one of "gd", "sgd", and "{}" got!'.format(solver)
        assert solve_dJ in ['debug'], \
            'solve_dJ must be one of "debug", and "{}" got!'.format(solve_dJ)
        self.alpha = alpha
        self.copy_X = copy_X
        self.normalize = normalize
        self.solver = solver
        self.solve_dJ = solve_dJ
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X, y):
        if self.copy_X:
            X = X.copy()
        if self.normalize:
            X = StandardScaler_().fit_transform(X)
        X_b = np.c_[np.ones((len(X), 1)), X]  # 拼接构造含有常数项b的新样本数据
        if self.solver == 'gd':
            self._fit_gd(X_b, y)
        else:
            pass
        return self

    def _fit_gd(self, X_b, y):  # 梯度下降法训练模型
        theta = np.ones(X_b.shape[1])
        for _ in range(self.n_iters):
            dJ = _dJ_gd_debug(theta, X_b, y, J='Lasso', alpha=self.alpha)
            theta -= self.learning_rate * dJ
            if np.linalg.norm(dJ) <= self.tol:  # 提前终止条件
                break
        self._theta = theta
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

    def predict(self, X_test):
        if self.copy_X:
            X_test = X_test.copy()
        if self.normalize:
            X_test = StandardScaler_().fit_transform(X_test)
        return np.c_[np.ones((len(X_test), 1)), X_test].dot(self._theta)

    def score(self, X_test, y_test, scoring=None):
        # score默认采取R2评价指标
        score_metrics = {'r2': r2_score_,
                         'mean_squared_error': mean_squared_error_
                         }
        y_pre = self.predict(X_test)
        if scoring:
            return score_metrics[scoring](y_test, y_pre)
        return r2_score_(y_test, y_pre)


class LogisticRegression_:
    """
    梯度下降法求导
    """

    def __init__(self, copy_X=True, normalize=True, n_iters=1000, alpha=0.01):
        self.copy_X = copy_X
        self.normalize = normalize
        self.n_iters = n_iters
        self.alpha = alpha
        self.theta_ = None
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, theta, X):
        return 1 / (1 + np.exp(-X.dot(theta)))

    def _dJ(self, theta, X, y):
        sigmoid = self._sigmoid(theta, X)
        return X.T.dot(sigmoid - y)

    def fit(self, X, y):
        if self.copy_X:
            X = X.copy()
        if self.normalize:
            X = StandardScaler_().fit_transform(X)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        theta = np.ones(X.shape[1])
        for _ in range(self.n_iters):
            theta -= self.alpha * self._dJ(theta, X, y)
        self.theta_ = theta
        self.coef_ = self.theta_[1:]
        self.intercept_ = self.theta_[0]
        return self

    def predict(self, X):
        return (self.predict_proba(X) > 0.5) * 1  # bool型转数值型

    def predict_proba(self, X):
        assert self.theta_ is not None, 'you should fit before predict'
        if self.copy_X:
            X = X.copy()
        if self.normalize:
            X = StandardScaler_().fit_transform(X)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self._sigmoid(self.theta_, X)

    def score(self, X, y):
        y_pre = self.predict(X)
        return accuracy_score_(y, y_pre)


def _dJ_gd_math(theta, X_b, y, J='LR', alpha=0):
    """
    数学公式求解梯度，通过alpha来兼容LR和Ridge模型
    仅适用于于LR和Ridge模型
    """
    assert J in ['LR', 'Ridge'], 'J should be one of "LR", "Ridge"'
    if J == 'LR':
        return X_b.T.dot(X_b.dot(theta) - y) / len(y)
    else:
        return X_b.T.dot(X_b.dot(theta) - y) / len(y) + alpha * theta


def J_lr(theta, X_b, y, alpha=0):
    """LR模型的损失函数"""
    return np.sum((X_b.dot(theta) - y) ** 2) / len(y)


def J_ridge(theta, X_b, y, alpha=1):
    """Ridge模型的损失函数"""
    return np.sum((X_b.dot(theta) - y) ** 2) / len(y) + alpha * np.sum(np.square(theta[1:]))  # 不包含theta0


def J_lasso(theta, X_b, y, alpha=0.1):
    """Lasso模型的损失函数"""
    return np.sum((X_b.dot(theta) - y) ** 2) / len(y) + alpha * np.sum(np.abs(theta[1:]))  # 不包含theta0


def _dJ_gd_debug(theta, X_b, y, J='LR', alpha=0):
    """
    根据损失函数变化求解梯度
    适用于LR、Ridge和Lasso
    通过J选择模型
    """
    assert J in ['LR', 'Ridge', 'Lasso'], 'J should be one of "LR", "Ridge", "Lasso"'
    J_funcs = {'LR': J_lr, 'Ridge': J_ridge, 'Lasso': J_lasso}
    J_func = J_funcs[J]
    dJ = np.empty(len(theta))
    DELT = 1e-4
    for i in range(len(theta)):
        theta1 = theta.copy()
        theta1[i] += DELT
        dJ[i] = (J_func(theta1, X_b, y, alpha) - J_func(theta, X_b, y, alpha)) / DELT  # dJ/dx = (J(x+delt) - J(x))/delt
    return dJ


def _dJ_sgd(theta, X_b, y, idx=None, J='LR', alpha=0):
    """利用单个样本特征向量求解梯度信息"""
    assert J in ['LR', 'Ridge'], 'J should be one of "LR", "Ridge"'
    if idx is None:
        idx = np.random.randint(len(y))
    if J == 'LR':
        return (X_b[idx].dot(theta) - y[idx]) * X_b[idx]
    else:
        return (X_b[idx].dot(theta) - y[idx]) * X_b[idx] + alpha * theta
