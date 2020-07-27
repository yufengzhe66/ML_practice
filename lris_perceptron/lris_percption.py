# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:51:16 2020

@author: xue
"""

#感知机：二分类模型  随机梯度下降
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt



#载入数据
"""
鸢尾花数据集是原则20世纪30年代的经典数据集。它是用统计进行分类的鼻祖。

sklearn包不仅囊括很多机器学习的算法，也自带了许多经典的数据集，鸢尾花数据集就是其中之一。
"""
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
y = np.array([1 if i==1 else -1 for i in y])

"""

#Perception
class Model:
    #初始化相应的参数
    def __init__(self):
        self.w = np.ones(len(data_[0])-1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1
        
        
        
    def sign(self,x,w,b):
        y = np.dot(x,w)+b
        return y
    
    
    def fit(self,X_train,y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X,self.w,self.b)<=0:
                    self.w += self.l_rate*np.dot(y,X)
                    self.b += self.l_rate*y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
        return 'Perception Model!'
    
    def score(self):
        pass
 


perception = Model()
perception.fit(X,y)


x_points = np.linspace(4,7,10)
y_ = -(perception.w[0]*x_points+perception.b)/perception.w[1]

plt.plot(x_points,y_)


plt.plot(data_[:50, 0], data_[:50, 1], 'bo', color='blue', label='0')
plt.plot(data_[50:100, 0], data_[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()


"""
from sklearn.linear_model import Perceptron
#clf = Perceptron(fit_intercept=True,max_iter=1000,shuffle=False)
clf = Perceptron( penalty=None, alpha=0.0001, fit_intercept=True, max_iter=1000, 
                 tol=0.001, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, 
                 random_state=0, early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, class_weight=None, warm_start=False)
clf.fit(X,y)


print(clf.coef_)
print(clf.intercept_)



x_points = np.arange(4,8)
y_=-(clf.coef_[0][0]*x_points + clf.intercept_)/clf.coef_[0][1]

plt.plot(x_points,y_)


plt.plot(data_[:50, 0], data_[:50, 1], 'bo', color='blue', label='0')
plt.plot(data_[50:100, 0], data_[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()




        
        
    
