'''
首先下载安装模块sklearn
其次安装 Graphviz（自行百度）
最好在ipython nootbook 中查看图形
'''
from sklearn import tree  # 加载决策树
from sklearn.feature_extraction import DictVectorizer  # 将特征值映射列表转换为向量。将数据序列化
from sklearn import preprocessing
import numpy as np
import pydotplus  # 作图
from IPython.display import Image  # 作图
import graphviz  # 作图

xiguashujuList = []  # 建立西瓜数据列表，优先存储格式为字典
xiguajieguoList = []
shujuTital = ['seze', 'gendi', 'qiaosheng', 'wenli',
              'qibu', 'chugan', 'midu', 'hantangliang']
shujuData = [['qinglv', 'quansuo', 'zhuoxiang', 'qingxi', 'aoxian', 'yinghua', 0.697, 0.460],
             ['wuhei', 'quansuo', 'chenmen', 'qingxi', 'aoxian', 'yinghua', 0.774, 0.376],
             ['wuhei', 'quansuo', 'zhuoxiang', 'qingxi', 'aoxian', 'yinghua', 0.634, 0.264],
             ['qinglv', 'quansuo', 'chenmen', 'qingxi', 'aoxian', 'yinghua', 0.608, 0.318],
             ['qianbai', 'quansuo', 'zhuoxiang', 'qingxi', 'aoxian', 'yinghua', 0.556, 0.215],
             ['qinglv', 'shaoquan', 'zhuoxiang', 'qingxi', 'shaoao', 'zhantie', 0.403, 0.237],
             ['wuhei', 'shaoquan', 'zhuoxiang', 'shaohu', 'shaoao', 'zhantie', 0.481, 0.149],
             ['wuhei', 'shaoquan', 'zhuoxiang', 'qingxi', 'shaoao', 'yinghua', 0.437, 0.211],
             ['wuhei', 'shaoquan', 'chenmen', 'shaohu', 'shaoao', 'yinghua', 0.666, 0.091],
             ['qinglv', 'yingting', 'qingcui', 'qingxi', 'pingtan', 'zhantie', 0.243, 0.267],
             ['qianbai', 'yingting', 'qingcui', 'mohu', 'pingtan', 'yinghua', 0.245, 0.057],
             ['qianbai', 'quansuo', 'zhuoxiang', 'mohu', 'pingtan', 'zhantie', 0.343, 0.099],
             ['qinglv', 'shaoquan', 'zhuoxiang', 'shaohu', 'aoxian', 'yinghua', 0.639, 0.161],
             ['qianbai', 'shaoquan', 'chenmen', 'shaohu', 'aoxian', 'yinghua', 0.657, 0.198],
             ['wuhei', 'shaoquan', 'zhuoxiang', 'qingxi', 'shaoao', 'zhantie', 0.360, 0.370],
             ['qianbai', 'quansuo', 'zhuoxiang', 'mohu', 'pingtan', 'yinghua', 0.593, 0.042],
             ['qinglv', 'quansuo', 'chenmen', 'shaohu', 'shaoao', 'yinghua', 0.719, 0.103]]
jieguoTital = ['haogua']
jieguoData = [
    '是', '是', '是', '是', '是', '是', '是', '是', '否', '否', '否', '否', '否', '否', '否', '否', '否'
]
# 生成数据列表字典
for x in range(0, len(shujuData)):
    xiguashujuList.append(dict(zip(shujuTital, shujuData[x])))
vecx = DictVectorizer()
dummyX = vecx.fit_transform(xiguashujuList).toarray()  # 映射索引的array格式
feature_names = vecx.get_feature_names()
vecx.inverse_transform(dummyX)
# 类别
lb = preprocessing.LabelBinarizer()  # 数据处理，数值二分化处理
dummyY1 = lb.fit_transform(jieguoData)
target_names1 = np.array(['good', 'bad'])

clf = tree.DecisionTreeClassifier(
)  # 创建一个分类器，entropy决定了用ID3算法   criterion="entropy"
clf = clf.fit(dummyX, dummyY1)  # 分类
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('iris.pdf')
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=target_names1,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
