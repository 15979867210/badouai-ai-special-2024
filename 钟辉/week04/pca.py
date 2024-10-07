import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris
import numpy as np

X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
for attr in X.T:
    print(attr)
mean = np.array([np.mean(attr) for attr in X.T]) #样本集的特征均值
print('样本集的特征均值:\n',mean)
centrX = []
centrX = X - mean
print(centrX)
ns = np.shape(centrX)[0]
print(np.shape(X))
print(ns)
C= np.dot(centrX.T, centrX)/(ns - 1)
a,b = np.linalg.eig(C)
print('样本集的协方差矩阵C的特征值:\n', a)
print('样本集的协方差矩阵C的特征向量:\n', b)
ind = np.argsort(-1*a)
print(ind)
UT = [b[:,ind[i]] for i in range(2)]
UTT = [b[:,2]]
U = np.transpose(UT)
print(UT)
print(UTT)
print(U)
Z=np.dot(X,U)
print(Z)
