from preprocessing_ import *
from model_selection_ import *
from linear_model_ import LinearRegression_, Ridge_, Lasso_, LogisticRegression_
from neighbors_ import KNeighborsClassifier_, KNeighborsRegressor_
from tree_ import DecisionTreeClassifier_, DecisionTreeRegressor_
from naive_bayes_ import GaussianNB_, MultinomialNB_
from cluster_ import KMeans_
from sklearn.datasets import load_diabetes, load_breast_cancer, load_boston, load_digits, make_blobs, load_iris
from metrics_ import *
from sklearn.metrics import *
from sklearn.cluster import KMeans

X_reg, y = load_diabetes(return_X_y=True)
X_reg_train, X_reg_test, y_train, y_test = train_test_split_(X_reg, y)

myLR_lsq = LinearRegression_()
myLR_lsq.fit(X_reg_train, y_train)
score_lsq = myLR_lsq.score(X_reg_test, y_test)
print('糖尿病预测，线性回归score: ', score_lsq)

myRidge = Ridge_(solver='lsqr')
myRidge.fit(X_reg_train, y_train)
score_ridge = myRidge.score(X_reg_test, y_test)
print('糖尿病预测，岭回归score: ', score_ridge)

myRidge_gd = Ridge_(solver='gd', alpha=0.01)
myRidge_gd.fit(X_reg_train, y_train)
score_ridge_gd = myRidge_gd.score(X_reg_test, y_test)
print('糖尿病预测，梯度下降岭回归score: ', score_ridge_gd)

myRidge_sgd = Ridge_(solver='sgd', alpha=0.01)
myRidge_sgd.fit(X_reg_train, y_train)
score_ridge_sgd = myRidge_sgd.score(X_reg_test, y_test)
print('糖尿病预测，随机梯度下降岭回归score: ', score_ridge_sgd)

myLasso = Lasso_(solver='gd', alpha=0.01)
myLasso.fit(X_reg_train, y_train)
score_lasso = myLasso.score(X_reg_test, y_test)
print('糖尿病预测，LASSO回归score: ', score_lasso)

knn_reg = KNeighborsRegressor_(weights='distance')
knn_reg.fit(X_reg_train, y_train)
score_knn_reg = knn_reg.score(X_reg_test, y_test)
print('糖尿病预测，KNN回归score: ', score_knn_reg)

X_clf, y_clf = load_iris(return_X_y=True)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split_(X_clf, y_clf)
knn = KNeighborsClassifier_()
knn.fit(X_clf_train, y_clf_train)
print('KNN鸢尾花分类评分: ', knn.score(X_clf_test, y_clf_test))
gnb = GaussianNB_()
gnb.fit(X_clf_train, y_clf_train)
print('高斯朴素贝叶斯鸢尾花分类评分: ', gnb.score(X_clf_test, y_clf_test))

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split_(X, y)
dt = DecisionTreeClassifier_(max_depth=10)
dt.fit(X_train, y_train)
print('决策树手写数字识别准确率：', dt.score(X_test, y_test))
mnb = MultinomialNB_()
mnb.fit(X_train, y_train)
print('多项式朴素贝叶斯手写数字识别准确率：', mnb.score(X_test, y_test))

Xr, yr = load_boston(return_X_y=True)
Xr_train, Xr_test, yr_train, yr_test = train_test_split_(Xr, yr)
dtr_ = DecisionTreeRegressor_(max_depth=3)
dtr_.fit(Xr_train, yr_train)
print('决策树波士顿房价回归评分：', dtr_.score(Xr_test, yr_test))

X, y = make_blobs(centers=4, n_samples=500, n_features=2, cluster_std=2)
X_train, X_test, y_train, y_test = train_test_split_(X, y)
km_ = KMeans_(n_clusters=4)
km_.fit(X_train)
print('自定义聚类轮廓系数', km_.score(X_test))
km = KMeans(n_clusters=4)
km.fit(X_train)
y_pre = km.predict(X_test)
print('系统KMeans系统轮廓系数', silhouette_score(X_test, y_pre))
print('系统KMeans自算轮廓系数', silhouette_score_(X_test, y_pre))

