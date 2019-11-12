# -*- coding:utf-8 -*-

# import warnings

import numpy as np

from algorithm.estimator import Estimator


# warnings.filterwarnings('error')


class SVDPlusPlus(Estimator):
    """
    属性
    ---------
    n_factors : 隐式因子数
    n_epochs : 迭代次数
    lr : 学习速率
    reg : 正则因子
    """
    
    def __init__(self, n_factors=20, n_epochs=10, lr=0.007, reg=0.002):
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
    
    def _train(self):
        self._prepare()
        for current_epoch in range(self.n_epochs):
            print(" processing epoch {}".format(current_epoch))
            self._iteration()
    
    def _prepare(self):
        self.train_dataset = self.train_dataset
        self.user_num = self.train_dataset.matrix.shape[0]
        self.item_num = self.train_dataset.matrix.shape[1]
        # global mean
        self.global_mean = self.train_dataset.global_mean
        # user bias
        self.bu = np.zeros(self.user_num, np.double)
        
        # item bias
        self.bi = np.zeros(self.item_num, np.double)
        
        # user factor
        self.p = np.random.normal(size=(self.user_num, self.n_factors)) + 0.1
        
        # item factor
        self.q = np.random.normal(size=(self.item_num, self.n_factors)) + 0.1
        
        # item preference factor
        self.y = np.random.normal(size=(self.item_num, self.n_factors)) + 0.1
    
    def _iteration(self):
        for u, i, r in self.train_dataset.all_ratings():
            # 用户u点评的item集
            Nu = self.train_dataset.get_user(u)[0]
            sqrt_Nu = np.sqrt(len(Nu))
            
            # 基于用户u点评的item集推测u的implicit偏好
            y_u = np.sum(self.y[Nu], axis=0)
            u_impl = y_u / sqrt_Nu
            # 预测值
            rp = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.q[i], self.p[u] + u_impl)
            # 误差
            e_ui = r - rp
            print(e_ui)
            try:
                self.bu[u] += self.lr * (e_ui - self.reg * self.bu[u])
                self.bi[i] += self.lr * (e_ui - self.reg * self.bi[i])
                self.p[u] += self.lr * (e_ui * self.q[i] - self.reg * self.p[u])
                self.q[i] += self.lr * (e_ui * (self.p[u] + u_impl) - self.reg * self.q[i])
                for j in Nu:
                    self.y[j] += self.lr * (e_ui * self.q[i] / sqrt_Nu - self.reg * self.y[j])
            except RuntimeWarning:
                print(RuntimeWarning)
                print(self.bu[u])
                print(self.bi[i])
                print(self.p[u])
                print(self.q[i])
                exit(0)
            except ValueError:
                print(ValueError)
                print(self.bu[u])
                print(self.bi[i])
                print(self.p[u])
                print(self.q[i])
                exit(0)
    
    def predict(self, u, i):
        Nu = self.train_dataset.get_user(u)[0]
        sqrt_Nu = np.sqrt(len(Nu))
        y_u = np.sum(self.y[Nu], axis=0)
        
        est = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.q[i], self.p[u] + y_u / sqrt_Nu)
        return est
