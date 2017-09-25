"""
    非线性分类器
    支持向量机
    SVC
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs
from tqdm import tqdm
# 产生数据的函数


def create_data(centers, num=100, std=0.7):
    # scikit中的make_blobs方法常被用来生成聚类算法的测试数据
    # http://datahref.com/archives/191
    x, labels_true = make_blobs(
        n_samples=num, centers=centers, cluster_std=std)
    return x, labels_true


centers = [[1, 1], [2, 3], [4, 5], [10, 20]]
x, labels_true = create_data(centers, 1000, 0.5)


# 测试 SVC 的用法。这里使用的是最简单的线性核
cls = svm.SVC(kernel='linear')
cls.fit(x, labels_true)
predicted_labels = cls.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=predicted_labels)
print('Score: %.2f' % cls.score(x, labels_true))
plt.show()


'''
测试多项式核的 SVC 的预测性能随 degree、gamma、coef0 的影响.
'''
# 测试 degree
degrees = range(1, 10)
train_scores = []
degrees_score = []
for degree in tqdm(degrees):
    cls = svm.SVC(kernel='poly', degree=degree)
    cls.fit(x, labels_true)
    train_scores.append(cls.score(x, labels_true))
    degrees_score.append(degree)


plt.scatter(degrees_score, train_scores)
plt.show()


# 测试 gamma ，此时 degree 固定为 3
gammas = range(1, 20)
train_scores = []
gammas_scores = []
for gamma in tqdm(gammas):
    cls = svm.SVC(kernel='poly', gamma=gamma, degree=1)
    cls.fit(x, labels_true)
    train_scores.append(cls.score(x, labels_true))
    gammas_scores.append(gamma)


plt.scatter(gammas_scores, train_scores)
plt.show()

# 测试 r ，此时 gamma固定为10 ， degree 固定为 3
rs = range(0, 20)
train_scores = []
r_scores = []
for r in tqdm(rs):
    cls = svm.SVC(kernel='poly', gamma=10, degree=1, coef0=r)
    cls.fit(x, labels_true)
    train_scores.append(cls.score(x, labels_true))
    r_scores.append(r)

plt.scatter(r_scores, train_scores)
plt.show()


'''
测试 高斯核的 SVC 的预测性能随 gamma 参数的影响
'''
gammas = range(1, 20)
train_scores = []
gamma_scores = []
for gamma in tqdm(gammas):
    cls = svm.SVC(kernel='rbf', gamma=gamma)
    cls.fit(x, labels_true)
    train_scores.append(cls.score(x, labels_true))
    gamma_scores.append(gamma)

plt.scatter(gamma_scores, train_scores)
plt.show()


'''
测试 sigmoid 核的 SVC 的预测性能随 gamma、coef0 的影响.
'''
# 测试 gamma ，固定 coef0 为 0
gammas = np.logspace(-2, 1)
train_scores = []
for gamma in tqdm(gammas):
    cls = svm.SVC(kernel='sigmoid', gamma=gamma, coef0=0)
    cls.fit(x, labels_true)
    train_scores.append(cls.score(x, labels_true))
plt.scatter(gammas, train_scores)
plt.show()

# 测试 r，固定 gamma 为 0.01
rs = np.linspace(0, 5)
train_scores = []
for r in tqdm(rs):
    cls = svm.SVC(kernel='sigmoid', coef0=r, gamma=0.01)
    cls.fit(x, labels_true)
    train_scores.append(cls.score(x, labels_true))
plt.scatter(rs, train_scores)
plt.show()
