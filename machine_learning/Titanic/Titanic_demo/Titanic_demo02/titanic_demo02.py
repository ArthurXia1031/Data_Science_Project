# -*- coding:utf-8 -*-
# @Time       :2022/11/5 & 2:51 PM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
# python中出现警告
# MatplotlibDeprecationWarning:
# The compare_versions function was deprecated in Matplotlib 3.2 and will be removed two minor releases later.
#
# 上述警告的意思是： matplotlib 3.2中已被弃用，将在两个较小的版本之后删除。如果还想用那么就不能升级版本
#
# 可以用
# import warnings
# warnings.filterwarnings(‘ignore’)
# 忽略显示


# https://zhuanlan.zhihu.com/p/71840687

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 设置1000列的时候才换行
pd.set_option('display.width', 1000)

train = pd.read_csv('train.csv')
print(train)

print(train.info())

print('----------------------DIVIDED-------------------------')

print(train.describe())

# describe 函数可以直接计算 data的各种 计量数据

#        PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare
# count   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000
# mean    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208
# std     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429
# min       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000
# 25%     223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400
# 50%     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200
# 75%     668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000
# max     891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200

print('----------------------DIVIDED-------------------------')

fig = plt.figure(figsize=(20, 8))

fig.set(alpha=0.5)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # 设置图例字体、位置、数值等等
        plt.text(rect.get_x(), 1.01 * height, '%s' %
                 float(height), size=11, family="Times new roman")


plt.subplot2grid((2, 3), (0, 0))

train.Survived.value_counts().plot(kind='bar')
plt.xticks(rotation=360)
plt.title('Survived Num')
plt.ylabel('Num')

plt.subplot2grid((2, 3), (0, 1))
train.Pclass.value_counts().plot(kind='bar')
plt.xticks(rotation=360)
plt.title('Pclass')
plt.ylabel('Num')

plt.subplot2grid((2, 3), (0, 2))
# age
plt.scatter(train.Survived, train.Age)
plt.ylabel('Age')
plt.grid(b=True, which='major', axis='y')
plt.title('Age range')

plt.subplot2grid((2, 3), (1, 0), colspan=2)
train.Age[train.Pclass == 1].plot(kind='kde')
train.Age[train.Pclass == 2].plot(kind='kde')
train.Age[train.Pclass == 3].plot(kind='kde')
plt.xlabel('Age')
plt.ylabel('Num')
plt.title('Range of Pclass')
plt.legend(('First_class', '2nd_class', '3rd_class'), loc='best')

plt.subplot2grid((2, 3), (1, 2))
train.Embarked.value_counts().plot(kind='bar')
plt.xticks(rotation=360)
plt.title('Embarked Num')
plt.ylabel('Num')

plt.show()

print('----------------------DIVIDED-------------------------')

# 图还是比数字好看多了。所以我们在图上可以看出来，被救的人300多点，不到半数；
# 3等舱乘客灰常多；遇难和获救的人年龄似乎跨度都很广；
# 3个不同的舱年龄总体趋势似乎也一致，2/3等舱乘客20岁多点的人最多，1等舱40岁左右的最多(→_→似乎符合财富和年龄的分配哈，咳咳，别理我，我瞎扯的)；
# 登船港口人数按照S、C、Q递减，而且S远多于另外俩港口。
#
# 这个时候我们可能会有一些想法了：
#
# 不同舱位/乘客等级可能和财富/地位有关系，最后获救概率可能会不一样 年龄对获救概率也一定是有影响的，
# 毕竟前面说了，副船长还说『小孩和女士先走』呢 和登船港口是不是有关系呢？
# 也许登船港口不同，人的出身地位不同？
#
# 口说无凭，空想无益。老老实实再来统计统计，看看这些属性值的统计分布吧。

print("""
五、属性与获救结果的关联
舱位等级与获救情况分析""")

Survived_0 = train.Pclass[train.Survived == 0].value_counts()
Survived_1 = train.Pclass[train.Survived == 1].value_counts()

df = pd.DataFrame({'Survived:': Survived_1, 'Unsurvived': Survived_0})
df.plot(kind='bar', stacked=True)
plt.xticks(rotation=360)
plt.title('Survived in different Level')
plt.xlabel('Level')
plt.ylabel('Num')

plt.show()

print('----------------------DIVIDED-------------------------')

# 性别与获救情况分析
Survived_m = train.Survived[train.Sex == 'male'].value_counts()
Survived_f = train.Survived[train.Sex == 'female'].value_counts()

df = pd.DataFrame({'Male:': Survived_m, 'Female': Survived_f})
df.plot(kind='bar', stacked=True)
plt.xticks(rotation=360)
plt.title('Survived in Sex Level')
plt.xlabel('Sex')
plt.ylabel('Num')

plt.show()

print('----------------------DIVIDED-------------------------')

print("""# 舱位等级和性别对获救影响的详细分析""")

# fig = plt.figure()
# fig.set(alpha=0.5)

fig = plt.figure(figsize=(20, 8))
fig.set(alpha=0.65)

plt.title('Survived Data about the Pclass & Sex')
# first title with the figure

ax1 = fig.add_subplot(1, 4, 1)
# same as add_subplot(141)

train.Survived[train.Sex == 'female'][train.Pclass != 3].value_counts().plot(
    kind='bar',
    label='female highclass',
    color='#FA2479'
)

ax1.set_xticklabels(['Survived', 'Unsurvived'], rotation=0)
ax1.legend(['Female/Highclass'], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
train.Survived[train.Sex == 'female'][train.Pclass == 3].value_counts().plot(
    kind='bar',
    label='female, low class',
    color='pink'
)
ax2.set_xticklabels(['Survived', 'Unsurvived'], rotation=0)
ax2.legend(['Female/lowclass'], loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)

train.Survived[train.Sex == 'male'][train.Pclass != 3].value_counts().plot(
    kind='bar',
    label='male, high class',
    color='lightblue'
)
ax3.set_xticklabels(['Survived', 'Unsurvived'], rotation=0)
ax3.legend(['Male/Highclass'], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
train.Survived[train.Sex == 'male'][train.Pclass == 3].value_counts().plot(
    kind='bar',
    label='male, low class',
    color='steelblue'
)
ax4.set_xticklabels(['Survived', 'Unsurvived'], rotation=0)
ax4.legend(['Male/lowclass'], loc='best')

plt.show()

print('----------------------DIVIDED-------------------------')

# print(train.Survived[train.Sex == 'male'][train.Pclass != 3])
# dataframe 判断语句

# 登船港口与获救情况分析

fig = plt.figure(figsize=(20, 8))
fig.set(alpha=0.2)

Survived_0 = train.Embarked[train.Survived == 0].value_counts()
Survived_1 = train.Embarked[train.Survived == 1].value_counts()
df = pd.DataFrame({'Survived': Survived_1, 'Unsurvived': Survived_0})

df.plot(kind='bar',
        stacked=True)

plt.title('Survived with each Embarked')
plt.xlabel('Embark')
plt.ylabel('Num of people')
plt.show()

print('----------------------DIVIDED-------------------------')

# number of family and influence of the surviving

Survived_0 = train.SibSp[train.Survived == 0].value_counts()
Survived_1 = train.SibSp[train.Survived == 1].value_counts()

df = pd.DataFrame({'Survived': Survived_1, 'Unsurvived': Survived_0})

# fig = plt.figure(figsize=(20, 8))
# ax1 = fig.add_subplot(141)

df.plot(kind='bar',
        stacked=True)

plt.title('Num of family with the Survived')
plt.xlabel('Num of famliy')
plt.ylabel('Num of survived')

plt.show()

print('----------------------DIVIDED-------------------------')

Survived_0 = train.Parch[train.Survived == 0].value_counts()
Survived_1 = train.Parch[train.Survived == 1].value_counts()

df = pd.DataFrame({'Survived': Survived_1, 'Unsurvived': Survived_0})

print(df)

df.plot(kind='bar',
        stacked=True)

plt.title('Children Num with survived')
plt.xlabel('Children Num')
plt.ylabel('Num')

plt.show()

# emmm，好像也看不出来啥问题啊。但是为啥还有父母与小孩个数为0的
# 还这么多都是，难道是孙悟空吗，从石头里面跳出来的？？看来这个数据有问题啊，问题还很大，算了算了，这个不考虑了

print('----------------------DIVIDED-------------------------')

fig = plt.figure()
fig.set(alpha=0.2)

Survived_cabin = train.Survived[pd.notnull(train.Cabin)].value_counts()
Survived_nocabin = train.Survived[pd.isnull(train.Cabin)].value_counts()

df = pd.DataFrame({'Survived': Survived_cabin, 'UnSurvived': Survived_nocabin})
df.plot(kind='bar',
        stacked=True)

plt.title('Survived with Cabin')
plt.xlabel('Cabin data')
plt.ylabel('Num')

plt.show()

print('----------------------DIVIDED-------------------------')

# 六、简单数据预处理
# 大体数据的情况看了一遍，对感兴趣的属性也有个大概的了解了。
#
# 下一步干啥？咱们该处理处理这些数据，为机器学习建模做点准备了。
#
#
#
# 对了，我这里说的数据预处理，其实就包括了很多Kaggler津津乐道的feature engineering过程，灰常灰常有必要！
# 『特征工程(feature engineering)太重要了！』
# 『特征工程(feature engineering)太重要了！』
# 『特征工程(feature engineering)太重要了！』

print("""# 先从最突出的数据属性开始吧，对，Cabin和Age，有丢失数据实在是对下一步工作影响太大""")

# 先说Cabin，暂时我们就按照刚才说的，按Cabin有无数据，将这个属性处理成Yes和No两种类型吧。

print(train.head(10))

print('----------------------DIVIDED-------------------------')


# 通常遇到缺值的情况，我们会有几种常见的处理方式
# 如果缺值的样本占总数比例极高，我们可能就直接舍弃了，作为特征加入的话，可能反倒带入noise，影响最后的结果了
# 如果缺值的样本适中，而该属性非连续值特征属性(比如说类目属性)，那就把NaN作为一个新类别，加到类别特征中
# 如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到属性类目中。
# 有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。
# 本例中，后两种处理方式应该都是可行的，我们先试试拟合补全吧(虽然说没有特别多的背景可供我们拟合，这不一定是一个多么好的选择)

def cabin_trans(value):
    if value == np.NAN:
        return 0
    else:
        return 1


train['Cabin'] = train['Cabin'].map(lambda x: cabin_trans(x))

print(train.head(10))

print('----------------------DIVIDED-------------------------')

# Age

# 我们这里用scikit-learn中的RandomForest来拟合一下缺失的年龄数据
# (注：RandomForest是一个用在原始数据中做不同采样，建立多颗DecisionTree，
# 再进行average等等来降低过拟合现象，提高结果的机器学习算法，我们之后会介绍到

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
