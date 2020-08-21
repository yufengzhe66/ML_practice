# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 23:26:42 2020

@author: xue
"""

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100,[0,1,-1]])
#    print(data)
    
    return data[:,:2],data[:,-1]
    
    
X,y=create_data()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

class LogisticRegressionClassifier:
    def __init__(self,max_iter=200,learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
    def sigmoid(self,x):
        return 1/(1+exp(-x))
    
    #为属性增加值为1的偏置项
    def data_matrix(self,X):
        data_mat = []
        for d in X:
            data_mat.append([1.0,*d])
        return data_mat
    
    def fit(self,X,y):
        #获取属性矩阵
        data_mat = self.data_matrix(X)
        #初始化权重矩阵
        self.weight = np.zeros((len(data_mat[0]),1), dtype=np.float32)
        
        for iter in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i],self.weight))
                error = y[i] - result
                self.weight += self.learning_rate*error*np.transpose([data_mat[i]])
        
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate, self.max_iter))
        
    
    def score(self,X_test,y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for i in range(len(X_test)):
            if(self.sigmoid(np.dot(X_test[i],self.weight))>0.5 and y_test[i]==1) or (self.sigmoid(np.dot(X_test[i],self.weight))<0.5 and y_test[i]==0):
                right += 1
                
        return right/len(X_test)
    
    
lr = LogisticRegressionClassifier()   
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))



#x_ponits = np.arange(4, 8)
#y_ = -(lr.weight[1]*x_ponits + lr.weight[0])/lr.weight[2]
#plt.plot(x_ponits, y_)
#
#plt.scatter(X[:50,0],X[:50,1], label='0')
#plt.scatter(X[50:,0],X[50:,1], label='1')
#plt.legend()


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=200)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

print(clf.coef_,clf.intercept_)

x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_points + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_points, y_)


plt.plot(X[:50, 0], X[:50, 1], 'bo', color='blue', label='0')
plt.plot(X[50:, 0], X[50:, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()




