# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 23:01:56 2020

@author: xue
"""

#最小二乘法


#第一步 ：导入相应的包，并生成数据
import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
#matplotlib inline


#定义一个需要拟合的函数
def real_func(x):
    return np.sin(2*np.pi*x)



#定义多项书
def fit_func(p,x):
    f=np.poly1d(p)
    return f(x)



#定义残差（损失函数）
def loss_func(p,x,y):
    ret = fit_func(p,x)-y
    return ret


#生成10个点
x = np.linspace(0,1,10)
x_points = np.linspace(0,1,1000)
y_ = real_func(x)
 
#为y加上正态分布的数据扰动（噪声）
y = [np.random.normal(0,0.1)+y1 for y1 in y_]


def fitting(M=0):
#随机初始化多项式参数
    p_init = np.random.rand(M+1)
    #最小二乘法
    p_lsq = leastsq(loss_func,p_init,args=(x,y))
    print('Fitting Parameters:',p_lsq[0])


    #可视化
    plt.plot(x_points,real_func(x_points),label='real')
    plt.plot(x_points,fit_func(p_lsq[0],x_points),label='fitted curve')
    plt.plot(x,y,'bo',label='noise')
    plt.legend()
    plt.show()
    return p_lsq

#欠拟合，其中M表示多项式的个数
p_lsq_0 = fitting(M=0)

  
p_lsq_1 = fitting(M=1)

#合适的拟合
p_lsq_3 = fitting(M=3)   

#过拟合
p_lsq_9 = fitting(M=9)
    
    
    
#定义正则化项
regularization = 0.0001


#正则化使用L2范数
def loss_func_regularizaition(p,x,y):
    ret = fit_func(p,x)-y
    ret = np.append(ret,np.sqrt(0.5*regularization*np.square(p)))
    return ret


p_init = np.random.rand(9+1)

p_lsq_regularization = leastsq(loss_func_regularizaition,p_init,args=(x,y))


plt.plot(x_points,real_func(x_points),label='real')
plt.plot(x_points,fit_func(p_lsq_9[0],x_points),label='fitted curve')
plt.plot(x_points,fit_func(p_lsq_regularization[0],x_points),label='regularizaition')
plt.plot(x,y,'bo',label='noise')
plt.legend()
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
