# -*- coding:utf-8 -*-
# @Time       :11/29/22 & 10:44 AM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains
# -*- encoding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('iris.pdf')

import pygraphviz as pgv
