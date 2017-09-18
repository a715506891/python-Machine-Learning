from sklearn import cluster
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn import naive_bayes
# 产生数据的函数


def create_data(centers, num=100, std=0.7):
    # scikit中的make_blobs方法常被用来生成聚类算法的测试数据
    # http://datahref.com/archives/191
    x, labels_true = make_blobs(
        n_samples=num, centers=centers, cluster_std=std)
    return x, labels_true


centers = [[1, 1], [2, 3], [4, 6], [10, 20]]
x, labels_true = create_data(centers, 1000, 0.5)
plt.scatter(x[:, 0], x[:, 1], c=labels_true)
plt.show()


# 高斯贝叶斯分类器
cls = naive_bayes.GaussianNB()
cls.fit(x, labels_true)
predicted_labels = cls.predict(x)
score = cls.predict_proba(x)
print('Training Score: %.2f' % cls.score(x, labels_true))
plt.scatter(x[:, 0], x[:, 1], c=predicted_labels)
plt.show()


# 多项式贝叶斯分类器
alphas = range(-2, 100)
train_scores = []
train_alpha = []
for alpha in alphas:
    cls = naive_bayes.MultinomialNB(alpha=alpha)
    cls.fit(x[(x[:, 0] > 0) & (x[:, 1] > 0), :], labels_true[
            (x[:, 0] > 0) & (x[:, 1] > 0)])  # x必须为非负数？？？？
    train_scores.append(cls.score(x[(x[:, 0] > 0) & (x[:, 1] > 0), :], labels_true[
        (x[:, 0] > 0) & (x[:, 1] > 0)]))
    train_alpha.append(alpha)
plt.scatter(train_alpha, train_scores)
plt.show()
# 展示不同alpha的得分
# 查找最优alpha
a = train_scores.index(max(train_scores))  # 最大值的索引
cls = naive_bayes.MultinomialNB(alpha=train_alpha[a])
cls.fit(x[(x[:, 0] > 0) & (x[:, 1] > 0), :], labels_true[
    (x[:, 0] > 0) & (x[:, 1] > 0)])  # x必须为非负数？？？？
predicted_labels = cls.predict(x[(x[:, 0] > 0) & (x[:, 1] > 0), :])
score = cls.predict_proba(x[(x[:, 0] > 0) & (x[:, 1] > 0), :])
print('Training Score: %.2f' % cls.score(x, labels_true))
plt.scatter(x[(x[:, 0] > 0) & (x[:, 1] > 0), 0], x[
            (x[:, 0] > 0) & (x[:, 1] > 0), 1], c=predicted_labels)
plt.show()


# 伯努利贝叶斯分类器
# 多项式贝叶斯分类器
alphas = np.logspace(-5, 2, num=200)
train_scores = []
train_alpha = []
for alpha in alphas:
    cls = naive_bayes.BernoulliNB(alpha=alpha)
    cls.fit(x, labels_true)
    train_scores.append(cls.score(x, labels_true))
    train_alpha.append(alpha)
plt.scatter(train_alpha, train_scores)
plt.show()
# 展示不同alpha的得分
# 查找最优alpha
a = train_scores.index(max(train_scores))  # 最大值的索引
cls = naive_bayes.MultinomialNB(alpha=train_alpha[a])
cls = naive_bayes.BernoulliNB()
cls.fit(x, labels_true)
predicted_labels = cls.predict(x)
score = cls.predict_proba(x)
print('Training Score: %.2f' % cls.score(x, labels_true))
plt.scatter(x[:, 0], x[:, 1], c=predicted_labels)
plt.show()

# 测试 BernoulliNB 的预测性能随 binarize 参数的影响
min_x = min(np.min(x), np.min(x)) - 0.1
max_x = max(np.max(x), np.max(x)) + 0.1
binarizes = np.linspace(min_x, max_x, endpoint=True, num=100)
train_scoresb = []
train_binarizeb = []
for binarize in binarizes:
    cls = naive_bayes.BernoulliNB(binarize=binarize)
    cls.fit(x, labels_true)
    train_scoresb.append(cls.score(x, labels_true))
    train_binarizeb.append(binarize)
plt.scatter(train_binarizeb, train_scoresb)
plt.show()

# 最佳BernoulliNB
b = train_scoresb.index(max(train_scoresb))  # 最大值的索引
cls = naive_bayes.BernoulliNB(
    alpha=train_alpha[a], binarize=train_binarizeb[b])
cls.fit(x, labels_true)
predicted_labels = cls.predict(x)
score = cls.predict_proba(x)
print('Training Score: %.2f' % cls.score(x, labels_true))
plt.scatter(x[:, 0], x[:, 1], c=predicted_labels)
plt.show()
