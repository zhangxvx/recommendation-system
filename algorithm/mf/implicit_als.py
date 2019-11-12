# -*- coding:utf-8 -*-

import numpy as np
from scipy import sparse

from algorithm.estimator import IterationEstimator


class ImplicitALS(IterationEstimator):
    """
    隐式交替最小二乘，果然不适合显式数据，表现很离谱
    
    属性
    ---------
    n_factors : 隐式因子数
    n_epochs : 迭代次数
    reg : 正则因子
    alpha : 隐式数据评分系数
    """
    
    def __init__(self, n_factors=20, n_epochs=10, reg=0.1, alpha=40):
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.alpha = alpha
    
    def alternative(self, X, Y, is_user):
        reg_I = self.reg * sparse.eye(self.n_factors)
        ids = self.train_dataset.uids if is_user else self.train_dataset.iids
        for k in ids:
            if is_user:
                ru = self.train_dataset.matrix.A[k]
            else:
                ru = self.train_dataset.matrix.A[:, k].T
            m = Y.shape[0]
            C = sparse.eye(m) + sparse.eye(m) * self.alpha * ru
            X[k] = np.linalg.solve(Y.T.dot(C.A).dot(Y) + reg_I, Y.T.dot(C.A).dot(ru))
    
    def _prepare(self):
        self.user_num = self.train_dataset.matrix.shape[0]
        self.item_num = self.train_dataset.matrix.shape[1]
        self.X = np.random.normal(size=(self.user_num, self.n_factors))
        self.Y = np.random.normal(size=(self.item_num, self.n_factors))
    
    def _iteration(self):
        self.alternative(self.X, self.Y, True)
        self.alternative(self.Y, self.X, False)
    
    def _pred(self):
        return np.dot(self.X, self.Y.T)
    
    def predict(self, u, i):
        est = np.dot(self.X[u, :], self.Y[i, :])
        return est
