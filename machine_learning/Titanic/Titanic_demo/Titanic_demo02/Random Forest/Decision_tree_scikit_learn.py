# -*- coding:utf-8 -*-
# @Time       :11/29/22 & 10:02 AM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
from graphviz import Digraph

X = [[0, 158], [0, 160], [0, 156], [0, 165], [0, 171], [0, 149], [0, 168], [0, 170], [0, 162], [1, 165], [1, 171],
     [1, 180], [1, 178], [1, 168], [1, 172], [1, 177], [1, 182], [1, 162]]
y = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]

# 对身高数据进行预处理, 使其与性别数据大小差不多
for i in X:
    i[1] = i[1] / 100

# 建立并训练决策树
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# 预测结果
print(clf.predict([[0, 1.55], [0, 1.67], [1, 1.67], [1, 1.79]]))

# 输出结果为:
# [0, 1, 0, 1]

# 我们可以使用 export_graphviz 导出器以 Graphviz 格式导出决策树

# dot_data = tree.export_graphviz(clf, out_file=None)

dot_data = tree.export_graphviz(clf)

# with open("test.gv", 'w') as f:
#     f.write(dot_data)

import pydotplus

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('iris_2.pdf')

from sklearn import tree
from sklearn.datasets import load_iris

# 载入sklearn中自带的数据Iris，构造决策树

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# 训练完成后，我们可以用 export_graphviz 将树导出为 Graphviz 格式
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
