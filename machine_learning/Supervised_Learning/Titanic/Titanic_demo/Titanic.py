# -*- coding:utf-8 -*-
# @Time       :2022/11/2 & 11:47 PM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# input train data
train = pd.read_csv('train.csv')

# input test data
test = pd.read_csv('test.csv')

print('train data:', train.shape, 'test data:', test.shape)

rowNum_train = train.shape[0]

rowNum_test = test.shape[0]

print('train 有多少行数据：', rowNum_train,
      ',test 有多少行数据：', rowNum_test)

# 合并数据集：

full = train.append(test, ignore_index=True)
print('合并之后的数据集：', full.shape)

# 数据清洗：
# 数据类型缺失处理：年龄（Age）、Fare（船票价格）里面有缺失数据，需要用平均值填补

# #对于数据类型，处理缺失值最简单的方法就是用平均数来填充缺失值
print('Before Cleaning:')
print(full.info())

# age
full['Age'] = full['Age'].fillna(round(full['Age'].mean(), 1))
full['Fare'] = full['Fare'].fillna(round(full['Fare'].mean(), 1))

print('----------------------DIVIDED-------------------------')

print('After Cleaning:')
print(full.info())

print('----------------------DIVIDED-------------------------')

# 登船港口（Embarked）里面数据总数是1307，只缺失了2条数据

# #登船港口（Embarked）：查看数据
# '''
# 出发地点：S=英国南安普顿Southampton
# 途径地点1：C=法国 瑟堡市Cherbourg
# 途径地点2：Q=爱尔兰 昆士敦Queenstown
# '''

print(full['Embarked'].head())

print('----------------------DIVIDED-------------------------')
# 分类变量Embarked，看下最常见的类别，用其填充
print(full['Embarked'].value_counts())
# S    914
# C    270
# Q    123
# Name: Embarked, dtype: int64

full['Embarked'] = full['Embarked'].fillna('S')

print(full.info())

print('----------------------DIVIDED-------------------------')

# as for the data of Cabin
# 1309 - 295 > there are 1014 Null values and percent is over 77%
# so we are going to add a class named Unknown

full['Cabin'] = full['Cabin'].fillna('U')

print(full.info())

# df = pd.DataFrame(full)
# df.to_csv('full.csv'.format(full), encoding="GBK")

print('----------------------DIVIDED-------------------------')

# 特征处理
# 查看数据类型，分为三种数据类型。并对类别数据处理：用数值代替类别，并用One-hot编码

# 1 特征提取
# 01 数值类型：直接使用
# 02 时间序列：转为 年月日
# 03 分类数据：简单的可用数值代替类别（比如 1-male， 0-female）

# 原始数据有N种类别的 用One-hot编码（虚拟变量/哑变量）

sex_mapDict = {'male': 1,
               'female': 0}

# map 函数， 对series每个数据应用自定义的函数计算

print(full['Sex'].head())

full['Sex'] = full['Sex'].map(sex_mapDict)

print(full['Sex'].head())

print('----------------------DIVIDED-------------------------')

# Embarked 通过get_dummies进行 one-hot编码 产生虚拟变量（dummy variables)
# 列名前缀是 Embarked

embarkedDf = pd.DataFrame()

"""
use get_dummies 进行one-hot编码，产生虚拟变量
"""

embarkedDf = pd.get_dummies(full['Embarked'], prefix='Embarked')
print(embarkedDf.head())

full.drop('Embarked', axis=1, inplace=True)
print(full.head(5))
# 通过embarked 记录 embarked
# 所以drop掉 embarked这一列

print('----------------------DIVIDED-------------------------')

# 客舱等级 也需要one-hot编码 使用get_dummies,列名前缀是Pclass

pclassDf = pd.DataFrame()

pclassDf = pd.get_dummies(full['Pclass'], prefix='Pclass')

print(pclassDf.head())

# 把one-hot产生的虚拟变量 添加到full数据集
full = pd.concat([full, pclassDf], axis=1)

full.drop('Pclass', axis=1, inplace=True)

print(full.head())

print('----------------------DIVIDED-------------------------')


# 把one-hot产生的虚拟变量 添加到full数据集

# 定义函数： 从姓名中获取头衔
# 直接用split会导致 dataframe里的数据变成列表
# 接下来用字典匹配时会发生错误，这里用join 将其重新变成字符串形式

def getTitle(name):
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]

    str3 = str2.strip()
    # 直接用split会导致dataframe里的数据变成列表，接下去用字典匹配时会发生错误，这里用join将其重新变成字符串形式
    str4 = "".join(str3)

    return str4


# 存放提取后的特征
titleDf = pd.DataFrame()
# map函数：对series每个数据应用自定义的函数计算
titleDf['Title'] = full['Name'].map(getTitle)
print(titleDf.head())

print('----------------------DIVIDED-------------------------')

'''
定义以下几种头衔类别：
Officer政府官员
Royalty王室（皇室）
Mr已婚男士
Mrs已婚妇女
Miss年轻未婚女子
Master有技能的人/教师
'''

title_mapDict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}

# map函数
titleDf['Title'] = titleDf['Title'].map(title_mapDict)

# 使用get_dummies进行one-hot编码

titleDf = pd.get_dummies(titleDf['Title'])
print(titleDf.head())
# Master  Miss  Mr  Mrs  Officer  Royalty
# 0       0     0   1    0        0        0
# 1       0     0   0    1        0        0
# 2       0     1   0    0        0        0
# 3       0     0   0    1        0        0
# 4       0     0   1    0        0        0

print('----------------------DIVIDED-------------------------')

# 添加one-hot编码产生的虚拟变量到full

full = pd.concat([full, titleDf], axis=1)

full.drop('Name', axis=1, inplace=True)

# df = pd.DataFrame(full)
# df.to_csv('full.csv'.format(full), encoding="GBK")

# 客舱号： 提取客舱类别

cabinDf = pd.DataFrame()

'''
客舱号的类别值是首字母，例如：
C85 类别映射为首字母C
'''

full['Cabin'] = full['Cabin'].map(lambda c: c[0])
print(cabinDf.head())

cabinDf = pd.get_dummies(full['Cabin'], prefix='Cabin')

print(cabinDf.head())

# 添加到full中
full = pd.concat([full, cabinDf], axis=1)

# delete cabin of full
full.drop('Cabin', axis=1, inplace=True)

print(full.head())

df = pd.DataFrame(full)
df.to_csv('full.csv'.format(full), encoding="GBK")

print('----------------------DIVIDED-------------------------')

# 建立家庭人数和家庭类别
# 存放家庭信息

familyDf = pd.DataFrame()

'''
家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己
（因为乘客自己也是家庭成员的一个，所以这里加1）
'''

familyDf['FamilySize'] = full['Parch'] + full['SibSp'] + 1

'''
家庭类别：
小家庭Family_Single：家庭人数=1
中等家庭Family_Small: 2<=家庭人数<=4
大家庭Family_Large: 家庭人数>=5
'''

# if 条件为真的时候返回if前面的内容 否则返回0

familyDf['Family_Single'] = familyDf['FamilySize'].map(lambda s: 1 if s == 1 else 0)
familyDf['Family_Small'] = familyDf['FamilySize'].map(lambda s: 1 if 2 <= s <= 2 else 0)
familyDf['Family_large'] = familyDf['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

print(familyDf.head())

full = pd.concat([full, familyDf], axis=1)

print('----------------------DIVIDED-------------------------')

# 相关系数法：
# 计算各个特征的相关系数

# 相关性矩阵
corrDf = full.corr()

print(corrDf)

#                PassengerId  Survived  ...  Family_Small  Family_large
# PassengerId       1.000000 -0.005007  ...     -0.035715     -0.063415
# Survived         -0.005007  1.000000  ...      0.163157     -0.125147
# Sex               0.013406 -0.543351  ...     -0.163546     -0.077748
# Age               0.025730 -0.070376  ...      0.090557     -0.161217
# SibSp            -0.055224 -0.035322  ...      0.125727      0.699681
# Parch             0.008942  0.081629  ...     -0.088528      0.624627
# Fare              0.031416  0.257307  ...      0.162190      0.170853
# Pclass_1          0.026495  0.285904  ...      0.212444     -0.067523
# Pclass_2          0.022714  0.093349  ...      0.011068     -0.118495
# Pclass_3         -0.041544 -0.322308  ...     -0.192890      0.155560
# Master            0.002254  0.085221  ...     -0.056199      0.301809
# Miss             -0.050027  0.332795  ...     -0.059877      0.083422
# Mr                0.014116 -0.549199  ...     -0.132618     -0.194207
# Mrs               0.033299  0.344935  ...      0.282682      0.012893
# Officer           0.002231 -0.031316  ...     -0.017106     -0.034572
# Royalty           0.004400  0.033391  ...      0.027195     -0.017542
# Cabin_A          -0.002831  0.022287  ...     -0.030189     -0.033799
# Cabin_B           0.015895  0.175095  ...      0.067172      0.013470
# Cabin_C           0.006092  0.114652  ...      0.170579      0.001362
# Cabin_D           0.000549  0.150716  ...      0.159358     -0.049336
# Cabin_E          -0.008136  0.145321  ...      0.053017     -0.046485
# Cabin_F           0.000306  0.057935  ...     -0.059729     -0.033009
# Cabin_G          -0.045949  0.016040  ...      0.003303     -0.016008
# Cabin_T          -0.023049 -0.026456  ...     -0.012934     -0.007148
# Cabin_U           0.000208 -0.316912  ...     -0.205042      0.056438
# FamilySize       -0.031437  0.016639  ...      0.034312      0.801623
# Family_Single     0.028546 -0.203367  ...     -0.577114     -0.318944
# Family_Small     -0.035715  0.163157  ...      1.000000     -0.120925
# Family_large     -0.063415 -0.125147  ...     -0.120925      1.000000


print('----------------------DIVIDED-------------------------')

corr_a = corrDf['Survived'].sort_values(ascending=False)

print(corr_a)

# Survived         1.000000
# Mrs              0.344935
# Miss             0.332795
# Pclass_1         0.285904
# Fare             0.257307
# Cabin_B          0.175095
# Family_Small     0.163157
# Cabin_D          0.150716
# Cabin_E          0.145321
# Cabin_C          0.114652
# Pclass_2         0.093349
# Master           0.085221
# Parch            0.081629
# Cabin_F          0.057935
# Royalty          0.033391
# Cabin_A          0.022287
# FamilySize       0.016639
# Cabin_G          0.016040
# PassengerId     -0.005007
# Cabin_T         -0.026456
# Officer         -0.031316
# SibSp           -0.035322
# Age             -0.070376
# Family_large    -0.125147
# Family_Single   -0.203367
# Cabin_U         -0.316912
# Pclass_3        -0.322308
# Sex             -0.543351
# Mr              -0.549199
# Name: Survived, dtype: float64

print('----------------------DIVIDED-------------------------')

"""
根据各个特征与生成情况（Survived）的相关系数大小
我们选择了这几个特征作为模型的输入：
头衔（前面所在的数据集titleDf
客舱等级（pclassDf）
家庭大小（familyDf）
船票价格（Fare）
船舱号（cabinDf）
登船港口（embarkedDf）
性别（Sex）
"""

# 特征选择

full_X = pd.concat([titleDf,
                    pclassDf,
                    familyDf,
                    full['Fare'],
                    cabinDf,
                    embarkedDf,
                    full['Sex'],
                    ], axis=1)

print(full_X.head())

print('----------------------DIVIDED-------------------------')

# 构建模型

# 建立训练数据集和测试数据集

# 我们使用Kaggle泰坦尼克号项目给的训练数据集，做为我们的原始数据集（记为source），
#
# 从这个原始数据集中拆分出训练数据集（记为train：用于模型训练）和测试数据集（记为test：用于模型评估）。
#
# full_X表示的是测试数据集和验证数据集合并在一起的数据。通过指定行号，把测试或者验证数据从合并以后（full）的数据里选出来
#
# sourceRow是我们在最开始合并数据前知道的，原始数据集有总共有891条数据
#
# 从特征集合full_X中提取原始数据集提取前891行数据时，我们要减去1，因为行号是从0开始的。

# 原始数据集有 891行

sourceRow = 891

# 原始数据集：特征
source_X = full_X.loc[0: sourceRow - 1, :]

# 原始数据集：标签

source_Y = full.loc[0:sourceRow - 1, 'Survived']

# 预测数据集：特征
pred_X = full_X.loc[sourceRow:, :]

"""
确保这里原始数据 取的是前891行的数据，不然后面模型会有错误"""

print('原始数据集有多少行:', source_X.shape[0])

# 预测数据集大小
print('预测数据集有多少行：', pred_X.shape[0])

# 从sklearn中 导入交叉验证中常用的函数 train_test_split:

print('----------------------DIVIDED-------------------------')

from sklearn.model_selection import train_test_split

# 建立模型的 数据集

train_X, test_X, train_Y, test_Y = train_test_split(source_X,
                                                    source_Y,
                                                    test_size=0.2,
                                                    train_size=0.8)

print('原始数据集特征：', source_X.shape,
      '训练数据集特征：', train_X.shape,
      '测试数据集特征：', test_X.shape)
print('原始数据集标签：', source_Y.shape,
      '训练数据集标签：', train_Y.shape,
      '测试数据集标签：', test_Y.shape)

# 选择 机器学习算法：

# 这里选择逻辑回归算法

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
# 迭代次数问题
# https://blog.csdn.net/pxyp123/article/details/124436727

model.fit(train_X, train_Y)

score = model.score(test_X, test_Y)

print(score)

# 0.8212290502793296

df = pd.DataFrame(full)
df.to_csv('full.csv'.format(full), encoding="GBK")
