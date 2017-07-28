import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline  # 简易模式
from sklearn import linear_model
from sklearn.pipeline import Pipeline  # 复杂模式


# 逐步拟合
# 最小二乘法

# PolynomialFeatures使用教程
# X = np.arange(6).reshape(3, 2)#生成3行两列随机数据
# X
# array([[0, 1],
#       [2, 3],
#       [4, 5]])
# poly = PolynomialFeatures(2)
# poly.fit_transform(X)
# array([[  1.,   0.,   1.,   0.,   0.,   1.],
#       [  1.,   2.,   3.,   4.,   6.,   9.],
#       [  1.,   4.,   5.,  16.,  20.,  25.]])
# poly = PolynomialFeatures(interaction_only=True)#是真只产生交叉数据
# poly.fit_transform(X)
# array([[  1.,   0.,   1.,   0.],
#       [  1.,   2.,   3.,   6.],
#       [  1.,   4.,   5.,  20.]])

# 定义原始函数x_plot = np.linspace(0, 10, 100)
x1 = np.linspace(8, 26, 100)  # x1 的取值
x2 = np.linspace(7, 15, 100)  # x2 的取值
x = [[x1[z], x2[z]]for z in range(0, len(x1))]
y = 10 + 2 * x1 - 2 * x2 + x1**2 - 2 * x2**2 + x1 * x2
poly = PolynomialFeatures(2)
xDeg = poly.fit_transform(x)
yxDeg = y
model = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model.fit(x, y)
model.named_steps  # 查看字段表达
a = model.named_steps['linearregression'].coef_
b = model.named_steps['linearregression'].intercept_
print(model.named_steps['linearregression'].coef_)  # 参数
print(model.named_steps['linearregression'].intercept_)  # 常量
print(model.named_steps['linearregression'].score(x, y))
y_plot = model.predict([9, 10])  # 测试数据
print(y_plot)
# 实验参数
x11 = 9
x22 = 10
y = 10 + 2 * x11 - 2 * x22 + x11**2 - 2 * x22**2 + x11 * x22

yTest = b + a[1] * x11 + a[2] * x22 + a[3] * \
    x11**2 + a[4] * x11 * x22 + a[5] * x22**2  # 按照PolynomialFeatures函数的顺序排列

# 一元多次
modelTest = Pipeline([('aa', PolynomialFeatures(degree=3)),
                      ('bb', linear_model.LinearRegression())])
x = np.arange(5)
y = 3 - 2 * x + x**2 - x**3
modelTest.fit(x[:, np.newaxis], y)
modelTest.named_steps['bb'].coef_
