# -*- coding:utf-8 -*-
# @Time       :11/28/22 & 6:54 PM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 设置1000列的时候才换行
pd.set_option('display.width', 1000)

df = pd.DataFrame({
    '账号': [714466, 714466, 714466, 737550, 146832, 218895, 218895, 412290, 740150, 141962, 163416, 239344, 239344,
             307599, 688981, 729833, 729833],
    '客户名称': ['华山派股份有限公司', '华山派股份有限公司', '华山派股份有限公司', '丐帮(北京) 合伙人公司',
                 '恶人谷资产管理公司', '桃花岛', '桃花岛', '有间客栈', '逍遥子影业',
                 '白驼山(上海)影视艺术有限公司', '聚贤庄', '全真教药业', '全真教药业', '天地会快递', '福寿堂',
                 '快手三教育培训有限公司', '快手三教育培训有限公司'],
    '销售': ['令狐冲', '令狐冲', '令狐冲', '令狐冲', '江小鱼', '江小鱼', '江小鱼', '段誉', '段誉', '欧阳克', '欧阳克',
             '欧阳克', '欧阳克', '韦小宝', '韦小宝', '韦小宝',
             '韦小宝'],
    '销售总监': ['岳不群', '岳不群', '岳不群', '岳不群', '岳不群', '岳不群', '岳不群', '岳不群', '岳不群', '完颜洪烈',
                 '完颜洪烈', '完颜洪烈', '完颜洪烈', '完颜洪烈',
                 '完颜洪烈', '完颜洪烈', '完颜洪烈'],
    '产品': ['黑玉断续膏', '葵花宝典', '含笑半步癫', '黑玉断续膏', '黑玉断续膏', '黑玉断续膏', '葵花宝典', '含笑半步癫',
             '黑玉断续膏', '黑玉断续膏', '黑玉断续膏', '含笑半步癫',
             '葵花宝典', '含笑半步癫', '黑玉断续膏', '黑玉断续膏', '如意勾'],
    '数量': [1, 2, 1, 3, 1, 3, 1, 2, 4, 2, 2, 1, 3, 5, 2, 3, 1],
    '价格': [3000, 2000, 1000, 3000, 1000, 3000, 1000, 2000, 4000, 2000, 2000, 1000, 2000, 3000, 1000, 4000, 2000],
    '状态': ['流程中', '流程中', '待审批', '驳回', '已完成', '流程中', '流程中', '待审批', '驳回', '已完成', '流程中',
             '待审批', '待审批', '已完成', '已完成', '驳回',
             '流程中'],
})

print(df)

print('----------------------DIVIDED-------------------------')

print(pd.pivot_table(df, index=['客户名称']))

print('----------------------DIVIDED-------------------------')

print(pd.pivot_table(df, index=['销售总监', '销售', '客户名称']))

# 所有index只是展示 非数据类 column 来表示 index

print('----------------------DIVIDED-------------------------')

# # **values**
#
# 如果不需要显示全部的数值列，可以用Values参数指定

print(pd.pivot_table(df, index=['销售总监', '销售'], values=['状态']))
# Empty DataFrame
# Columns: []
# Index: [(完颜洪烈, 欧阳克), (完颜洪烈, 韦小宝), (岳不群, 令狐冲), (岳不群, 段誉), (岳不群, 江小鱼)]

print('----------------------DIVIDED-------------------------')

print(pd.pivot_table(df, index=['销售总监', '销售'], values=['数量']))

print('----------------------DIVIDED-------------------------')
# # **aggfunc**
#
# 当我们未设置aggfunc时，它默认aggfunc='mean’计算均值。

print(pd.pivot_table(df, index=['销售总监', '销售', '客户名称'], values=['数量', '价格'], aggfunc=np.sum))
print('----------------------DIVIDED-------------------------')

print(
    pd.pivot_table(df, index=['销售总监', '销售', '客户名称'], values=['数量', '价格'], aggfunc=[np.mean, len, np.sum]))

print('----------------------DIVIDED-------------------------')

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=matplotlib.cm.get_cmap(name="bwr"), alpha=0.7)
plt.grid(True)
plt.show()
