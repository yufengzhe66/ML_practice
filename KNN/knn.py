# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 19:15:22 2020

@author: xue
"""

#K近邻算法：非常简单的算法
"""
p=1 曼哈顿距离
p=2 欧式距离
p=3 闵氏距离
"""
import math
from itertools import combinations
#计算距离
def L(x,y,p=2):
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i]-y[i]),p)
        return math.pow(sum,1/p)
    else:
        return 0


x1 = [1,1]
x2 = [5,1]
x3 = [4,4]


for i in range(1,5):
    r = {'1-{}'.format(c):L(x1,c,p=i)for c in[x2,x3]}
#    print(r)
    print(min(zip(r.values(),r.keys())))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter


data = load_iris()
data_fig = pd.DataFrame(data.data,columns=data.feature_names)
data_fig['label'] = data.target

data_fig.columns = ['sepal length','sepal width','petal length','petal width','label']
data_fig.label.value_counts()

"""
plt.scatter(data_fig[:50]['sepal length'],data_fig[:50]['sepal width'],label='0')
plt.scatter(data_fig[50:100]['sepal length'],data_fig[50:100]['sepal width'],label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()

"""

data_ = np.array(data_fig.iloc[:100,[0,1,-1]])
X,y = data_[:,:-1],data_[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2)



#KNN算法不需要训练
class KNN:
    def __init__(self,X_train,y_train,n_neighbors=3,p=2):
        #n_neighbors表示需要选取临近点的个数
        #p表示距离度量
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train
    
    
    def predict(self,X):
        #取出n个点
        knn_list = []
        #选取前n个点，并计算距离
        for i in range(self.n):
            dist = np.linalg.norm(X-self.X_train[i], ord=self.p)
            knn_list.append((dist,self.y_train[i]))
            
        #knn_list是一个二维数组
        #从第N+1个点开始，和最大距离点进行比较，若小则替换掉  
        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list,key=lambda x:x[0]))
            dist = np.linalg.norm(X-self.X_train[i],ord=self.p)
            if knn_list[max_index][0]>dist:
                knn_list[max_index] = (dist, self.y_train[i])
        
#        print(knn_list)
        #统计
        #采用少数服从多数的原则进行表决
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
#        print(count_pairs)
        max_count = sorted(count_pairs,key=lambda x:x)[-1]
#        print(max_count)
        return max_count
    
    
    def score(self,X_test,y_test):
        right_count = 0
 
        for X,y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count/len(X_test)


clf = KNN(X_train,y_train)
print(clf.score(X_test,y_test))


test_point = [6.0,3.0]
print("Test Point:{}".format(clf.predict(test_point)))


plt.scatter(data_fig[:50]['sepal length'], data_fig[:50]['sepal width'], label='0')
plt.scatter(data_fig[50:100]['sepal length'],data_fig[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()




#调用scikitlearn内置的函数，实现knn算法

"""
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')  
#内置函数的默认参数

n_neighbors: 临近点个数
p: 距离度量
algorithm: 近邻算法，可选{'auto', 'ball_tree', 'kd_tree', 'brute'}
weights: 确定近邻的权重
"""


from sklearn.neighbors import KNeighborsClassifier

clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train,y_train)
print(clf_sk.score(X_test,y_test))



