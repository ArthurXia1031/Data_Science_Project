# -*- coding:utf-8 -*-
# @Time       :2022/11/4 & 9:26 PM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains
import numpy as np
import pandas as pd

# 导入数据
# 训练数据

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

print('train data is ', train.shape,
      'test data is ', test.shape)

rowNum_train = train.shape[0]
rowNum_test = test.shape[0]

print("train data's row:", rowNum_train)
print("test data's row:", rowNum_test)

full = train.append(test)
# 合并数据集

print(full.info())

# 1309行
# 进行数据清洗


print('----------------------DIVIDED-------------------------')

# age
full['Age'] = full['Age'].fillna(full['Age'].mean())

# 5   Age          1309 non-null   float64

# fare
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())

# print(full.info())
#  9   Fare         1309 non-null   float64

# Embarked

print(full['Embarked'].value_counts())
# S    914
# C    270
# Q    123
# Name: Embarked, dtype: int64

# replace with S

full['Embarked'] = full['Embarked'].fillna('S')

# print(full.info())
#  11  Embarked     1309 non-null   object

# Cabin
# 船舱号（Cabin）里面数据总数是295，缺失了1309-295=1014，缺失率=1014/1309=77.5%，
# 缺失比较大。所以新增了一个类U-未知

full['Cabin'] = full['Cabin'].fillna('U')

# print(full.info())
#  10  Cabin        1309 non-null   object
# 数据清洗完成

print('----------------------DIVIDED-------------------------')

# 数据分析及可视化

# 数据集提供了10个与乘客有关的数据特征，这里以数据特征为分析维度，探讨各个数据特征与幸存率的关系。
#
# 其中，3个数据特征不作分析，原因如下：
#
# Ticket（票号）：无法分类，没有参考价值
# Fare（票价）：票价由客舱等级决定，不必重复分析
# Cabin（客舱号）：缺失值数量太多，没有分析价值

# 其他7个数据特征中，Pclass（客舱等级）、Sex（性别）和Embarked（登船港口）可直接用于分析，
#
# 另外4个则需进行分组归类后，才能进一步分析
#
# 因此，幸存率分析维度如下：

print('----------------------DIVIDED-------------------------')

# 幸存者与家庭类别之间的关系

# 存放家庭信息
familyDf = pd.DataFrame()

familyDf['FamilySize'] = train['Parch'] + train['SibSp'] + 1


def familyGroup(fs):
    if fs == 1:
        return 'Family_Single'
    elif 2 <= fs <= 4:
        return 'Family_Small'
    else:
        return 'Family_Large'


familyDf['FamilyCategory'] = familyDf['FamilySize'].map(familyGroup)

print(familyDf.head())
#    FamilySize FamilyCategory
# 0           2   Family_Small
# 1           2   Family_Small
# 2           1  Family_Single
# 3           2   Family_Small
# 4           1  Family_Single
print('----------------------DIVIDED-------------------------')

train = pd.concat([train, familyDf], axis=1)

print(train.head(5))
#    PassengerId  Survived  Pclass  ... Embarked FamilySize  FamilyCategory
# 0            1         0       3  ...        S          2    Family_Small
# 1            2         1       1  ...        C          2    Family_Small
# 2            3         1       3  ...        S          1   Family_Single
# 3            4         1       1  ...        S          2    Family_Small
# 4            5         0       3  ...        S          1   Family_Single

print('----------------------DIVIDED-------------------------')

# 统计各类型家人的幸存人数

FamilyCgdf = pd.pivot_table(train,
                            index='FamilyCategory',
                            columns='Survived',
                            values='PassengerId',
                            aggfunc='count')

print(FamilyCgdf)
# Survived          0    1
# FamilyCategory
# Family_Large     52   10
# Family_Single   374  163
# Family_Small    123  169

print('----------------------DIVIDED-------------------------')

# 汇总各类型家庭的幸存比例

# div函数用法1：除以同一个值
print(FamilyCgdf.div(10))
# Survived           0     1
# FamilyCategory
# Family_Large     5.2   1.0
# Family_Single   37.4  16.3
# Family_Small    12.3  16.9

print('----------------------DIVIDED-------------------------')

# div函数用法2：根据不同索引，除以不同值

otherS = pd.Series([10, 100, 1000], index=['Family_Large', 'Family_Single', 'Family_Small'])
print(FamilyCgdf.div(otherS, axis='index'))
# Survived            0      1
# FamilyCategory
# Family_Large    5.200  1.000
# Family_Single   3.740  1.630
# Family_Small    0.123  0.169

print(otherS)
# Family_Large       10
# Family_Single     100
# Family_Small     1000

print('----------------------DIVIDED-------------------------')

# 汇总统计家庭类别与是否幸存的比例
FamilyCgdf2 = FamilyCgdf.div(FamilyCgdf.sum(axis=1), axis=0)

print(FamilyCgdf2)
# Survived               0         1
# FamilyCategory
# Family_Large    0.838710  0.161290
# Family_Single   0.696462  0.303538
# Family_Small    0.421233  0.578767

print('----------------------DIVIDED-------------------------')

# 根据上面的数据框分别表示各个家庭的死亡率和幸存率，这里只获取幸存率。

# FamilyCgdf_rate = FamilyCgdf2['1']
#
# print(FamilyCgdf_rate)

# 区分 iloc 和 loc函数的区别
# iloc
# 取指定单行多列

FamilyCgdf_rate = FamilyCgdf2.iloc[:, 1]
print(FamilyCgdf_rate)
# FamilyCategory
# Family_Large     0.161290
# Family_Single    0.303538
# Family_Small     0.578767
# Name: 1, dtype: float64

print('----------------------DIVIDED-------------------------')

# print(FamilyCgdf2.loc[:,1])
# FamilyCategory
# Family_Large     0.161290
# Family_Single    0.303538
# Family_Small     0.578767
# Name: 1, dtype: float64

# loc
# 取第指定单行行，多列，与iloc一样

print('----------------------DIVIDED-------------------------')

# 分析结果数据可视化

import matplotlib.pyplot as plt

fig = plt.figure(1)
plt.figure(figsize=(12, 4))

ax1 = plt.subplot(1, 2, 1)
# subplot是将多个图画到一个平面上的工具。
# 其中，m表示是图排成m行，n表示图排成n列，也就是整个figure中有n个图是排成一行的，一共m行，如果m=2就是表示2行图。p表示图所在的位置，p=1表示从左到右从上到下的第一个位置。
# 因此subplot(2,2,1)表示在本区域里显示2行2列个图像，最后的1表示本图像显示在第一个位置。
# https://baijiahao.baidu.com/s?id=1714871501318340512&wfr=spider&for=pc

FamilyCgdf.plot(ax=ax1,
                kind='bar',
                stacked=True,
                color=['red', 'royalblue'])

plt.xticks(rotation=360)
plt.xlabel('Family')
plt.ylabel('Num')

plt.title('Family and Survived Num')

plt.legend(labels=['Not Survived', 'Survived'], loc='upper right')

ax2 = plt.subplot(1, 2, 2)
FamilyCgdf_rate.plot(ax=ax2, kind='bar', color='orange')

plt.xticks(rotation=360)
plt.xlabel('Family')
plt.ylabel('Survived Rate')

plt.title('Family and Survived Rate')

plt.show()

# 由上述可视化数据可知：
#
# 在人数方面，单身人士（Family_Single)总人数最多，其次是小家庭（Family_Small)，最少的是大家庭（Family_Large)；
#
# 在幸存率方面，小家庭（Family_Small)的幸存率最高，其次是单身人士（Family_Single)，人数最少的大家庭（Family_Large)幸存率最低。

print('----------------------DIVIDED-------------------------')


# 探寻 幸存率与头衔之间的关系

# 从姓名中获取头衔

def getTitle(name):
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]

    str3 = str2.strip()

    return str3


titleDf = pd.DataFrame()

titleDf['Title'] = train['Name'].map(getTitle)

print(titleDf['Title'].value_counts())

print('----------------------DIVIDED-------------------------')

# 建立 头衔映射关系

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

titleDf['Title'] = titleDf['Title'].map(title_mapDict)

print(titleDf.value_counts())
# Title
# Mr         517
# Miss       184
# Mrs        127
# Master      40
# Officer     18
# Royalty      5
# dtype: int64

print('----------------------DIVIDED-------------------------')

train = pd.concat([train, titleDf], axis=1)

print(train.head(5))

print('----------------------DIVIDED-------------------------')

TitleDf = pd.pivot_table(train,
                         index='Title',
                         columns='Survived',
                         values='PassengerId',
                         aggfunc='count')

print(TitleDf.head())
# Survived    0    1
# Title
# Master     17   23
# Miss       55  129
# Mr        436   81
# Mrs        26  101
# Officer    13    5

print('----------------------DIVIDED-------------------------')

# 汇总统计不同头衔的幸存率

TitleDf2 = TitleDf.div(TitleDf.sum(axis=1), axis=0)

print(TitleDf2)
# Survived         0         1
# Title
# Master    0.425000  0.575000
# Miss      0.298913  0.701087
# Mr        0.843327  0.156673
# Mrs       0.204724  0.795276
# Officer   0.722222  0.277778
# Royalty   0.400000  0.600000
print('----------------------DIVIDED-------------------------')

# 获取不同头衔的幸存率

TitleDf_rate = TitleDf2.iloc[:, 1]
print(TitleDf_rate)
# Master     0.575000
# Miss       0.701087
# Mr         0.156673
# Mrs        0.795276
# Officer    0.277778
# Royalty    0.600000
# Name: 1, dtype: float64

print('----------------------DIVIDED-------------------------')
# 分析可视化


fig = plt.figure(1)
plt.figure(figsize=(12, 4))
# 创建画纸（子图）
# 创建画纸，并选择画纸1
ax1 = plt.subplot(1, 2, 1)
# 在画纸1绘制堆积柱状图
TitleDf.plot(ax=ax1,  # 选择画纸1
             kind='bar',  # 选择图表类型
             stacked=True,  # 是否堆积
             color=['orangered', 'royalblue']  # 设置图表颜色
             )
plt.xticks(rotation=360)
plt.xlabel('Title')
plt.ylabel('Num')
plt.title('Title and Survived Num')
plt.legend(labels=['Not Survived', 'Survived'], loc='upper right')
ax2 = plt.subplot(1, 2, 2)
TitleDf_rate.plot(ax=ax2, kind='bar', color='orange')
plt.xticks(rotation=360)
plt.xlabel('Title')
plt.ylabel('Survived Rate')
plt.title('Title and Survived Rate')
plt.show()

# 由上述可视化数据可知：
# 在人数方面，头衔分类中人数最多的是已婚男士，未婚女士和已婚女士次之，其他头衔的只占少数；
# 幸存率方面，已婚男士最低，政府官员也较低，已婚女士和未婚女士的幸存率最高。


print('----------------------DIVIDED-------------------------')


# 4 根据年龄对应幸存率分析

def ageCut(a):
    if a < 13:
        return 'Children'
    elif 13 < a <= 30:
        return 'Youth'
    elif 30 < a <= 60:
        return 'Middle-aged'
    else:
        return 'The old'


train['AgeCategory'] = train['Age'].map(ageCut)

print(train[['AgeCategory', 'Age']].head(5))

#    AgeCategory   Age
# 0        Youth  22.0
# 1  Middle-aged  38.0
# 2        Youth  26.0
# 3  Middle-aged  35.0
# 4  Middle-aged  35.0

print('----------------------DIVIDED-------------------------')

AgeDf = pd.pivot_table(train,
                       index='AgeCategory',
                       values='PassengerId',
                       columns='Survived',
                       aggfunc='count',
                       fill_value=0)

print(AgeDf)

# Survived       0    1
# AgeCategory
# Children      29   40
# Middle-aged  164  119
# The old      142   59
# Youth        214  124

print('----------------------DIVIDED-------------------------')

# 汇总统计不同年龄段的幸存率
AgeDf2 = AgeDf.div(AgeDf.sum(axis=1), axis=0)

print(AgeDf2)
# Survived            0         1
# AgeCategory
# Children     0.420290  0.579710
# Middle-aged  0.579505  0.420495
# The old      0.706468  0.293532
# Youth        0.633136  0.366864

print('----------------------DIVIDED-------------------------')

AgeDf_rate = AgeDf2.iloc[:, 1]
print(AgeDf_rate)
# AgeCategory
# Children       0.579710
# Middle-aged    0.420495
# The old        0.293532
# Youth          0.366864
# Name: 1, dtype: float64

print('----------------------DIVIDED-------------------------')

# 分析数据可视化

fig = plt.figure(1)
plt.figure(figsize=(12, 4))

ax1 = plt.subplot(1, 2, 1)

AgeDf.plot(ax=ax1,
           kind='bar',
           stacked=True,
           color=['orangered', 'royalblue'])

plt.xticks(rotation=360)
plt.xlabel('Age')
plt.ylabel('Num')
plt.title('Age & Survived Num')

plt.legend(labels=['Not Survived', 'Survived'], loc='best')

ax2 = plt.subplot(1, 2, 2)
AgeDf_rate.plot(ax=ax2,
                kind='bar',
                stacked=True,
                color='orange')

plt.xticks(rotation=360)
plt.xlabel('Age')
plt.ylabel('Survived Rate')
plt.title('Age & Survived Rate')

plt.show()

# 由上述可视化数据可知：
# 在人数方面，青年人数最多，中年人次之，老年人最少；
# 在幸存率方面，儿童的幸存率最高，中年人次之，老年人最低。

print('----------------------DIVIDED-------------------------')

# 幸存率 与 客舱等级之间的关系

PclassDf = pd.pivot_table(train,
                          columns='Survived',
                          values='PassengerId',
                          index='Pclass',
                          aggfunc='count')

print(PclassDf)
# Survived    0    1
# Pclass
# 1          80  136
# 2          97   87
# 3         372  119

print('----------------------DIVIDED-------------------------')

PclassDf2 = PclassDf.div(PclassDf.sum(axis=1), axis=0)
print(PclassDf2)
# Survived         0         1
# Pclass
# 1         0.370370  0.629630
# 2         0.527174  0.472826
# 3         0.757637  0.242363
print('----------------------DIVIDED-------------------------')

Pclass_rate = PclassDf2.iloc[:, 1]
print(Pclass_rate)
# Pclass
# 1    0.629630
# 2    0.472826
# 3    0.242363

print('----------------------DIVIDED-------------------------')

# 分析结果可视化

fig = plt.figure(1)
plt.figure(figsize=(12, 4))

ax1 = plt.subplot(1, 2, 1)

PclassDf.plot(ax=ax1,
              kind='bar',
              stacked=True,
              color=['orangered', 'royalblue'])

plt.xticks(rotation=360)
plt.xlabel('Pclass')
plt.ylabel('Num')

plt.title('Pclass & Survived Num')

plt.legend(labels=['Not Survived', 'Survived'], loc='best')

ax2 = plt.subplot(1, 2, 2)

Pclass_rate.plot(ax=ax2,
                 kind='bar',
                 color='orange')

plt.xticks(rotation=360)
plt.xlabel('Pclass')
plt.ylabel('Survived Rate')

plt.title('Pclass and Survived Rate')

plt.show()

# 由上述可视化数据可知：
#
# 在人数方面， 三等舱的人数最多，其次是一等舱，最少的是二等舱，但一等舱与二等舱人数相差不大；
#
# 在幸存率方面，一等舱幸存率最高，二等舱次之，三等舱最低。


print('----------------------DIVIDED-------------------------')

SexDf = pd.pivot_table(train,
                       index='Sex',
                       columns='Survived',
                       values='PassengerId',
                       aggfunc='count')
print(SexDf)

# 汇总统计不同性别与是否幸存的比例
SexDf2 = SexDf.div(SexDf.sum(axis=1), axis=0)
print(SexDf2)

# 获取不同性别的幸存率
SexDf_rate = SexDf2.iloc[:, 1]
print(SexDf_rate)
print('----------------------DIVIDED-------------------------')

fig = plt.figure(1)
plt.figure(figsize=(12, 4))
ax1 = plt.subplot(1, 2, 1)
SexDf.plot(ax=ax1,  # 选择画纸1
           kind='bar',  # 选择图表类型
           stacked=True,  # 是否堆积
           color=['orangered', 'royalblue']  # 设置图表颜色
           )
plt.xticks(rotation=360)
plt.xlabel('Sex')
plt.ylabel('Num')
plt.title('Sex and Survived Num')
plt.legend(labels=['Not Survived', 'Survived'], loc='upper left')

ax2 = plt.subplot(1, 2, 2)
SexDf_rate.plot(ax=ax2, kind='bar', color='orange')
plt.xticks(rotation=360)
plt.xlabel('Sex')
plt.ylabel('Survived Rate')
plt.title('Sex and Survived Rate')
plt.show()

# 由上述可视化数据可知：
# 在人数方面， 男性人数最多；
# 在幸存率方面，女性则远远高于男性。

print('----------------------DIVIDED-------------------------')

EmbarkedDf = pd.pivot_table(train,
                            index='Embarked',
                            columns='Survived',
                            values='PassengerId',
                            aggfunc='count')
print(EmbarkedDf)

EmbarkedDf2 = EmbarkedDf.div(EmbarkedDf.sum(axis=1), axis=0)
print(EmbarkedDf2)

EmbarkedDf_rate = EmbarkedDf2.iloc[:, 1]
print(EmbarkedDf_rate)

fig = plt.figure(1)
plt.figure(figsize=(12, 4))
ax1 = plt.subplot(1, 2, 1)
EmbarkedDf.plot(ax=ax1,  # 选择画纸1
                kind='bar',  # 选择图表类型
                stacked=True,  # 是否堆积
                color=['orangered', 'royalblue']  # 设置图表颜色
                )
plt.xticks(rotation=360)
plt.xlabel('Embarked')
plt.ylabel('Num')
plt.title('Embarked and Survived Num')
plt.legend(labels=['Not Survived', 'Survived'], loc='upper left')

ax2 = plt.subplot(1, 2, 2)
EmbarkedDf_rate.plot(ax=ax2, kind='bar', color='orange')
plt.xticks(rotation=360)
plt.xlabel('Embarked')
plt.ylabel('Survived Rate')
plt.title('Embarked and Survived Rate')
plt.show()

# 由上述可视化数据可知：
# 在人数方面， 乘客绝大部分都是从Southampton登船，从Cherbourg登船的乘客次之，从Queenstown登船的乘客人数最少；
# 在幸存率方面，幸存率最高的是从Cherbourg登船的乘客，而人数最多的从Southampton登船的乘客幸存率最低。


# 总结
# 1.家庭类别：
#
# 在人数方面，单身人士最多，然而人数最少的大家庭幸存率最低，小家庭的幸存率最高。
#
# 2.头衔：
#
# 已婚男士人数最多，但幸存率最低，未婚女士和已婚女士人数人数虽然只有已婚男士的一半不到，幸存率却排在前两位，其中已婚女士幸存率最高。
#
# 3.年龄：
#
# 乘客主要以青年人为主，但儿童的幸存率最高，说明当时逃生时儿童优先；老年人的人数少，幸存率也低，可能是由于老人行动不便，来不及逃生或者是主动放弃了生存的机会。
#
# 4.客舱等级：
#
# 三等舱的人数最多，但幸存率从一等舱到三等舱依次下降，客舱等级越高，幸存率越高，说明面对灾难时，上层阶级逃机率最大。
#
# 5.性别：
#
# 在人数方面，男性乘客人数大约是女性乘客的两倍，但是男性乘客的幸存率比女性低很多，不及女性的三分之一，说明当时女士优先的原则深入人心。
#
# 6.登船港口：
#
# 在人数方面，三个港口上船的乘客中，来自Southampton港口的最多，可能是泰坦尼克号出发港口的原因；但是于Southampton港口登船的乘客幸存率最低；而人数不及来自Southampton港口三分之一的从Cherbourg登船的乘客幸存率最高，可能是因为该部分乘客船舱等级较高或者是以女性居多等因素。
#
# 综上，如果当时有一个小女孩，在父母陪伴下从Cherbourg港口登船，且乘坐的是一等舱，那么她从那次海难中幸存的概率最大；反之，带着一大家子亲戚从Southampton港口登船，乘坐在三等舱的男性老人，能够幸存的概率则最小。
