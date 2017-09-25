from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# 产生数据的函数


def create_data(centers, num=100, std=0.7):
    # scikit中的make_blobs方法常被用来生成聚类算法的测试数据
    # http://datahref.com/archives/191
    x, labels_true = make_blobs(
        n_samples=num, centers=centers, cluster_std=std)
    return x, labels_true


centers = [[1, 1], [2, 3], [4, 5], [10, 20]]
x, labels_true = create_data(centers, 1000, 0.5)


# 直接运行
cls = svm.LinearSVC()
cls.fit(x, labels_true)
predicted_labels = cls.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=predicted_labels)
print('Score: %.2f' % cls.score(x, labels_true))
plt.show()


'''
测试 LinearSVC 的预测性能随损失函数的影响
:param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
:return:  None
'''
# squared_hinge更高
losses = ['hinge', 'squared_hinge']
for loss in losses:
    cls = svm.LinearSVC(loss=loss)
    cls.fit(x, labels_true)
    print("Loss:%s" % loss)
    print('Score: %.2f' % cls.score(x, labels_true))


'''
测试 LinearSVC 的预测性能随正则化形式的影响
:param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
:return:  None
'''
# 都一样
L12 = ['l1', 'l2']
for p in L12:
    cls = svm.LinearSVC(penalty=p, dual=False)
    cls.fit(x, labels_true)
    print("penalty:%s" % p)
    print('Score: %.2f' % cls.score(x, labels_true))


'''
测试 LinearSVC 的预测性能随参数 C 的影响
:param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
:return:   None
'''
Cs = np.logspace(-2, 5)
train_scores = []
for C in Cs:
    cls = svm.LinearSVC(C=C)
    cls.fit(x, labels_true)
    train_scores.append(cls.score(x, labels_true))
    print("penalty:%s" % C)
    print('Score: %.2f' % cls.score(x, labels_true))


# 最佳分类
clst = svm.LinearSVC(C=3.72759372031, loss='squared_hinge')
clst.fit(x, labels_true)
predicted_labels = clst.predict(x)
print('Score: %.2f' % clst.score(x, labels_true))
plt.scatter(x[:, 0], x[:, 1], c=predicted_labels)
plt.show()
