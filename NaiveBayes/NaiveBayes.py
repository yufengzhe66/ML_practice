# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:52:18 2020

@author: xue
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math


def create_data():
    data = load_iris()
    df = pd.DataFrame(data.data,columns=data.feature_names)
    df['label'] = data.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data_ = np.array(df.iloc[:100,:])
    return data_[:,:-1],data_[:,-1]



X,y = create_data()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#print(X_test[0],y_test[0])
#print(sum(X)/len(X))

class NaiveBayes:
    def __init__(self):
        self.model = None
    
    #数学期望
    @staticmethod
    def mean(X):
        return sum(X)/float(len(X))
    
    
    #标准差（方差）
    def stdev(self,X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x-avg,2) for x in X])) / float(len(X))
    
    #概率密度函数
    def gaussian_probability(self,x,mean,stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1/(math.sqrt(2*math.pi)*stdev)) * exponent
    
    #处理X_train
    def summarize(self,train_data):
        summaries = [(self.mean(i),self.stdev(i)) for i in zip(*train_data)]
        return summaries
    
    
    #分别每一个分类的计算数学期望和标准差
    def fit(self,X,y):
        labels = list(set(y))
        data = {label:[]for label in labels}
        for f,label in zip(X,y):
            data[label].append(f)
        self.model = {label:self.summarize(value)for label,value in data.items()}
        return 'gaussianNB train done'
    
    
    #计算输入数据分到所有类的概率，并最终将每个特征值的概率相乘
    def calculate_probabilities(self,input_data):
        probabilities = {}
        for label,value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i],mean,stdev)
        return probabilities
    
    
    #类别
    def predict(self,X_test):
        label = sorted(self.calculate_probabilities(X_test).items(),key=lambda x:x[-1])[-1][0]
        return label
    
    def score(self,X_test,y_test):
        right = 0
        for X,y in zip(X_test,y_test):
            label = self.predict(X)
            if label == y:
                right += 1
        
        return right/float(len(X_test))
    
    
    
model = NaiveBayes()
print(model.fit(X_train,y_train))   
        
        
print(model.predict([4.4, 3.2, 1.3, 0.2]))      
            
print(model.score(X_test,y_test))    
        
        
        
        
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB

clf = GaussianNB()
print(clf.fit(X_train,y_train))

print(clf.score(X_test,y_test))

#print(clf.predict([4.4,3.2,1.3,0.2]))



clf = BernoulliNB()
print(clf.fit(X_train,y_train))

print(clf.score(X_test,y_test))

    
        
     
clf = MultinomialNB()
print(clf.fit(X_train,y_train))

print(clf.score(X_test,y_test))        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    