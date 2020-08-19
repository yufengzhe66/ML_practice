# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:04:08 2020

@author: xue
"""

#ID3 （基于信息增益）
#C4.5 （基于信息增益比）
#CART (gini指数)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
#
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math
from math import log
import pprint


#载入数据

def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels


#datasets, labels = create_data()
#
#train_data = pd.DataFrame(datasets, columns=labels)
#
#print(train_data)


#ID3算法
#计算熵：根据香农的熵计算公式，计算每个分类的熵
#label_count是一个dict
def calc_ent(datasets):
    data_len = len(datasets)
    label_count = {}
    for item in datasets:
        label = item[-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum([(p/data_len)*log(p/data_len,2) for p in label_count.values()])
    return ent


#经验条件熵:计算指定属性作为分类条件的熵
def calc_condent(datasets, axis=0):
    data_len = len(datasets)
    feature_set = {}
    for i in range(data_len):
        feature = datasets[i][axis]
        if feature not in feature_set:
            feature_set[feature] = []
        feature_set[feature].append(datasets[i])
    cond_ent = sum([(len(p)/data_len)*calc_ent(p) for p in feature_set.values()])
    return cond_ent

#计算信息增益
def info_gain(ent,cond_ent):
    return ent - cond_ent



#计算最佳的分类属性，即寻找每个属性作为分类依据的信息增益
def info_gain_train(datasets):
    feature_count = len(datasets[0]) - 1
    ent = calc_ent(datasets)
    best_feature = []
    for c in range(feature_count):
        c_info_gain = info_gain(ent,calc_condent(datasets,c))
        best_feature.append([c,c_info_gain])
        print('特征({}) - info_gain - {:.3f}'.format(labels[c],c_info_gain))
    #选取最大值
    best = max(best_feature, key=lambda x:x[1])
    return '特征({})的信息增益最大，选择为根节点特征'.format(labels[best[0]])
    

#print(info_gain_train(datasets))



#利用ID3算法生成决策树

class Node:
    def __init__(self,root=True,label=None,feature_name=None,feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        #设置返回值的标签
        self.result = {'label:': self.label, 'feature': self.feature, 'tree': self.tree}
        
    
    def __repr__(self):
        return '{}'.format(self.result)
        

    def add_node(self,val,node):
        self.tree[val] = node


    def predict(self,features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class DTree:
    def __init__(self,epsilon=0.1):
        self.epsilon = epsilon
        self.tree = {}
        
        
    @staticmethod
    def calc_ent(datasets):
        data_len = len(datasets)
        label_count = {}
        for item in datasets:
            label = item[-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p/data_len)*log(p/data_len,2) for p in label_count.values()])
        return ent


#经验条件熵:计算指定属性作为分类条件的熵
    def cond_ent(self,datasets, axis=0):
        data_len = len(datasets)
        feature_set = {}
        for i in range(data_len):
            feature = datasets[i][axis]
            if feature not in feature_set:
                feature_set[feature] = []
            feature_set[feature].append(datasets[i])
        cond_ent = sum([(len(p)/data_len)*self.calc_ent(p) for p in feature_set.values()])
        return cond_ent

#计算信息增益
    @staticmethod
    def info_gain(ent,cond_ent):
        return ent - cond_ent



#计算最佳的分类属性，即寻找每个属性作为分类依据的信息增益
    def info_gain_train(self,datasets):
        feature_count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(feature_count):
            c_info_gain = self.info_gain(ent,self.cond_ent(datasets,c))
            best_feature.append((c,c_info_gain))
#            print('特征({}) - info_gain - {:.3f}'.format(labels[c],c_info_gain))
        #选取最大值
        best = max(best_feature, key=lambda x:x[1])
        return best




    """
    input:数据集D(DataFrame格式)，特征集A，阈值eta
    output:决策树T
    """
    
    def train(self,train_data):
        _, y_train, features = train_data.iloc[:, :-1],train_data.iloc[:,-1],train_data.columns[:-1]
        
        if len(y_train.value_counts())==1:  #只属于一个类时
            return Node(root=True,label=y_train.iloc[0])
        
        if len(features)==0:        #只剩一个属性时，选取个数多的类为分类结果
            return Node(root=True,label=y_train.value_counts().sort_values(ascending=False).index[0])

        max_feature,max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]
        
        if max_info_gain < self.epsilon:
            return Node(root=True,label=y_train.value_counts().sort_values(ascending=False).index[0])
        
        #构建子集
        node_tree = Node(root=False,feature_name=max_feature_name,feature=max_feature)
        
        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name]==f].drop([max_feature_name],axis=1)
            
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f,sub_tree)
            
        return node_tree
    
    
    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)


datasets, labels = create_data()
data_df = pd.DataFrame(datasets, columns=labels)

dt = DTree()

tree = dt.fit(data_df)
#print(tree)

print(dt.predict(['老年', '否', '否', '一般']))



from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
import graphviz



def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:,:2], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



clf = DecisionTreeClassifier()
clf.fit(X_train, y_train,)

print(clf.score(X_test, y_test))

tree_pic = export_graphviz(clf, out_file="mytree.pdf")
with open('mytree.pdf') as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)


