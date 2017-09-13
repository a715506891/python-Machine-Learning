# 参考链接http://blog.csdn.net/xuanyuansen/article/details/41309033
# 高斯分布即正态分布
# 高斯混合聚类通过概率模型来表示聚类原型。
# 若已知高斯混合分布，则高斯混合聚类的原理是，如过样本x最有可能是z=k产生的，则将样本划分到本簇中。
# 个人理解：就是样本分布属于每各高斯分布的概率，去概率最大的划分到该簇。
# 示例代码
# -*- coding: utf-8 -*-
"""
    聚类和EM算法
    ~~~~~~~~~~~~~~~~
    GMM
    :copyright: (c) 2016 by the huaxz1986.
    :license: lgpl-3.0, see LICENSE for more details.
"""
from sklearn import mixture
from sklearn.metrics import adjusted_rand_score
from sklearn import cluster
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# 产生数据的函数


def create_data(centers, num=100, std=0.7):
    # scikit中的make_blobs方法常被用来生成聚类算法的测试数据
    # http://datahref.com/archives/191
    x, labels_true = make_blobs(
        n_samples=num, centers=centers, cluster_std=std)
    return x, labels_true


def test_GMM(*data):
    '''
    测试 GMM 的用法
    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X, labels_true = data
    clst = mixture.GaussianMixture()  # 默认值为1
    clst.fit(X)
    predicted_labels = clst.predict(X)
    print("ARI:%s" % adjusted_rand_score(labels_true, predicted_labels))
    print(predicted_labels)


centers = [[1, 1], [2, 2], [1, 2], [10, 20]]
x, labels_true = create_data(centers, 1000, 0.5)
test_GMM(x, labels_true)


def test_GMM_n_components(*data):
    '''
    测试 GMM 的聚类结果随 n_components 参数的影响
    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X, labels_true = data
    nums = range(1, 50)
    ARIs = []
    for num in nums:
        clst = mixture.GaussianMixture(n_components=num)
        clst.fit(X)
        predicted_labels = clst.predict(X)
        ARIs.append(adjusted_rand_score(labels_true, predicted_labels))

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(nums, ARIs, marker="+")
    ax.set_xlabel("n_components")
    ax.set_ylabel("ARI")
    fig.suptitle("GMM")
    plt.show()


test_GMM_n_components(x, labels_true)  # 根据不同聚类点，算出不同的ari指标


def test_GMM_cov_type(*data):
    '''
    测试 GMM 的聚类结果随协方差类型的影响
    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X, labels_true = data
    nums = range(1, 50)

    cov_types = ['spherical', 'tied', 'diag', 'full']
    markers = "+o*s"
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i, cov_type in enumerate(cov_types):
        ARIs = []
        for num in nums:
            clst = mixture.GaussianMixture(
                n_components=num, covariance_type=cov_type)
            clst.fit(X)
            predicted_labels = clst.predict(X)
            ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        ax.plot(nums, ARIs, marker=markers[
                i], label="covariance_type:%s" % cov_type)

    ax.set_xlabel("n_components")
    ax.legend(loc="best")
    ax.set_ylabel("ARI")
    fig.suptitle("GMM")
    plt.show()


test_GMM_cov_type(x, labels_true)
