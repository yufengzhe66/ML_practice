import os
import random
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
#from tensorflow import keras
from keras import utils as np_utils


def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float") #channel last
        Y = np.array(Y)
    return X, Y





def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,2):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr[1:201], Ytr[1:201], Xte[1:51], Yte[1:51]
	





	
def data_process():
    ROOT = 'cifar-10-batches-py'
    X_train,y_train,X_test,y_test = load_CIFAR10(ROOT)
    #数据归一化（data normalization）
    X_train /= 255
    X_test /= 255
    y_train = np_utils.to_categorical(y_train, num_classes = 10) 
    y_test = np_utils.to_categorical(y_test, num_classes = 10)    
    #数据类型转换，即压缩
    X_train = X_train.astype(np.float32) #'float32'占4个字节
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.uint8) # 'uint8'占1个字节
    y_test = y_test.astype(np.uint8)
    #分割训练集
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=0)
    
    
        
    return X_train,X_val,y_train,y_val,X_test,y_test
 


    
