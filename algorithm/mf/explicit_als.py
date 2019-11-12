# -*- coding:utf-8 -*-

import numpy as np
from scipy import sparse

from algorithm.estimator import IterationEstimator


class ExplicitALS(IterationEstimator):
    """
    显式交替最小二乘，算法表现一般，从它的损失函数也可以看出，是最
    简单的svd。只不过ALS相比SGD速度快一点, 一般10次迭代就能收敛
    
    属性
    ---------
    n_factors : 隐式因子数
    n_epochs : 迭代次数
    reg : 正则因子
    """
    
    def __init__(self, n_factors=20, n_epochs=10, reg=0.1):
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
    
    # 交替最小二乘法
    def alternative(self, X, Y, is_user):
        reg_I = self.reg * sparse.eye(self.n_factors)
        ids = self.train_dataset.uids if is_user else self.train_dataset.iids
        for k in ids:
            if is_user:
                action_idx = self.train_dataset.get_user(k)[0]
                r = self.train_dataset.matrix.A[k]
            else:
                action_idx = self.train_dataset.get_item(k)[0]
                r = self.train_dataset.matrix.A[:, k].T
            Y_u = Y[action_idx]
            X[k] = np.linalg.solve(np.dot(Y_u.T, Y_u) + reg_I, np.dot(Y.T, r))
            # X[k] = np.linalg.solve(np.dot(Y.T, Y) + reg_I, np.dot(Y.T, r))
    
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


if __name__ == '__main__':
    from util.matrix import Matrix
    
    data = sparse.csc_matrix(np.random.randint(0, 5, (5, 5)))
    print(data)
    ex = ExplicitALS()
    ex.train(Matrix(data))
    print(data.A)
    print(ex._pred())
