# TiTanic

##  环境

使用到的包可以通过`requirements.txt`进行查看，在虚拟环境中安装可以通过执行下面语句安装所需的包

```python
pip install -r requirements.txt -i https://pypi.douban.com/simple
```

**（更换国内镜像可以极大的提高包的安装效率）**

## 数据整理

### 查看数据集

数据集保存在`dataset`中，通过构造一个类DataDeal来处理相关的操作，在构造方法中载入数据集

```python
def __init__(self):
        self.df_train = pd.read_csv('../../dataset/train.csv')
        self.df_test = pd.read_csv('../../dataset/test.csv')
        self.data_clean = self.df_train.copy(deep=True)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        sns.set(font='SimHei', style='whitegrid', context='talk')
        self.clean_data = pd.read_csv('../../dataset/data_clean.csv')
        mpl.rcParams['axes.unicode_minus'] = False
```

**注意**：为了避免通过`matplotlib`或`seaborn`做图时，部分字体无法显示，可通过设置字体进行解决，即

```python
plt.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
```

`SimHei`保存在dataset中

### 数据清洗

由于数据集中有部分数据的缺失，所以要先将缺失的数据值进行填充，填充可以有多种方法，这里采用的是平均数、众数、中位数进行填充。数据填充完成后，通过 

```py
 print(self.clean_data.isnull().sum())
```

查看是否还有缺失值没有填充。

### 特征选择

这里是通过生成数据集的相关图进行参考来选择特征的。`def correllogram(self):`

![](https://gitee.com/cnpolaris-tian/giteePagesImages/raw/master/null/Titanic数据相关图.png)

最后选定```features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']```作为特征

##  模型训练(未完待续) 

> 决策数是一种无参监督学习方法，其目的是创建一种模型从数据特征中学习简单的决策规则来预测一个目标变量的值。

Titanic数据集小，数据规范化相对优秀，所以通过决策树实现Titanic简单预测相对简单。

需要预测的数据是`Survived`，即通过选定的一些特征来预测一位乘客在船上遇见事故的生还情况。

```python
train_features = train_data[features] # 训练数据
train_labels = train_data['Survived'] # 训练目标数据
test_features = test_data[features] # 测试数据
```

```python
dvec = DictVectorizer(sparse=False) # 特征向量化 sparse=False 意思是不产生稀疏矩阵0
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
# 构造 ID3 决策树
clf = DecisionTreeClassifier(criterion='entropy')
```

