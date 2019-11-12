# -*- coding:utf-8 -*-

import numpy as np
from scipy.sparse import lil_matrix

from algorithm.estimator import Estimator
from util.databuilder import DataBuilderPlus
from util.logger import logger


class UserCfPre(Estimator):
    """
    属性
    ---------
    min_coo : 有效交互数下限
    top_k : 相似矩阵topk
    """
    
    def __init__(self, min_coo=2, top_k=20, use_baseline=True):
        super().__init__()
        self.min = min_coo
        self.top_k = top_k
        self.weight = np.array([0.4, 0.3, 0.2, 0.1])
    
    def compute_cosine_similarity(self, user_num, item_num, users_ratings):
        sim = lil_matrix((user_num, user_num), dtype=np.double)
        dot = lil_matrix((user_num, user_num), dtype=np.double)  # 点积
        sq = np.zeros(user_num, dtype=np.double)  # 向量平方和
        coo = lil_matrix((user_num, user_num), dtype=np.double)  # 共现矩阵
        cur = 1
        for i, (u, r) in users_ratings:
            cur = cur + 1
            for k1 in range(len(u)):
                sq[u[k1]] += r[k1] ** 2
                for k2 in range(k1 + 1, len(u)):
                    u1, u2 = u[k1], u[k2]
                    dot[u1, u2] += r[k1] * r[k2]
                    coo[u1, u2] += 1
            self.progress(cur, item_num)
        
        # dok_matrix不适合进行矩阵算术操作，转为csc格式
        dot = dot.tocsc()
        coo = coo.tocsc()
        # 交互数低于限制全部清零
        dot.data[coo.data < self.min] = 0
        # 只需要考虑非0点积
        row, col = dot.nonzero()
        # cosine相似矩阵
        sim[row, col] = dot[row, col] / np.sqrt(sq[row] * sq[col])
        sim[col, row] = sim[row, col]
        return sim
    
    def compute_attr_similarity(self, user_num, user_data):
        user_data = np.array(user_data)
        sim = lil_matrix((user_num, user_num), dtype=np.double)  # 属性相似度
        for k1 in range(user_num):
            for k2 in range(k1 + 1, user_num):
                a = user_data[k1][1:] == user_data[k2][1:]
                sim[k1, k2] = (a * self.weight).sum()
        return sim  # 转化为数组
    
    def compute_sigmoid_similarity(self, user_num, item_num, users_ratings):
        sim = lil_matrix((user_num, user_num), dtype=np.double)
        sig = lil_matrix((user_num, user_num), dtype=np.double)  # sigmoid相似和
        coo = lil_matrix((user_num, user_num), dtype=np.int)  # 共现矩阵
        uoo = np.zeros(user_num, dtype=np.int)  # 用户评分数目
        cur = 1
        for i, (u, r) in users_ratings:
            cur = cur + 1
            for k1 in range(len(u)):
                uoo[u[k1]] += 1
                for k2 in range(k1 + 1, len(u)):
                    u1, u2 = u[k1], u[k2]
                    exp = np.exp(-abs(r[k1] - r[k2]))
                    sig[u1, u2] += 2 * exp / (1 + exp)
                    coo[u1, u2] += 1
            self.progress(cur, item_num)
        # dok_matrix不适合进行矩阵算术操作，转为csc格式
        sig = sig.tocsc()
        coo = coo.tocsc()
        # 交互数低于限制全部清零
        sig.data[coo.data < self.min] = 0
        # 只需要考虑非0相似度和
        row, col = sig.nonzero()
        # cosine相似矩阵
        sim[row, col] = sig[row, col] / coo[row, col]
        return sim, coo
    
    def compute_mixed_similarity(self, user_num):
        exp = np.exp(-self.coo.toarray())
        beta = 2 * exp / (1 + exp)
        sim = beta * self.attr_sim.toarray() + (1 - beta) * self.sigmoid_sim.toarray()
        
        # 另一种计算方式，使用稀疏矩阵
        # sim = lil_matrix((user_num, user_num), dtype=np.double)
        # for k1 in range(user_num):
        #     for k2 in range(k1 + 1, user_num):
        #         exp = np.exp(-self.coo[k1, k2])
        #         beta = 2 * exp / (1 + exp)
        #         sim[k1, k2] = beta * self.attr_sim[k1, k2] + (1 - beta) * self.sigmoid_sim[k1, k2]
        
        row, col = sim.nonzero()
        sim[col, row] = sim[row, col]
        return sim
    
    def _train(self):
        user_num = self.train_dataset.matrix.shape[0]
        item_num = self.train_dataset.matrix.shape[1]
        self.item_means = self.train_dataset.get_item_means()
        self.user_means = self.train_dataset.get_user_means()
        
        self.cos_sim = self.compute_cosine_similarity(user_num, item_num, self.train_dataset.get_items())
        self.attr_sim = self.compute_attr_similarity(user_num, self.user_dataset)
        self.sigmoid_sim, self.coo = self.compute_sigmoid_similarity(user_num, item_num, self.train_dataset.get_items())
        self.sim = self.compute_mixed_similarity(user_num)
        logger.debug('cos sim:')
        print(self.cos_sim.A)
        logger.debug('attr sim:')
        print(self.attr_sim.A)
        logger.debug('mixed sim')
        print(self.sim)
    
    def predict(self, u, i):
        ll, rr = self.train_dataset.get_item(i)  # ([列下标],[评分])
        neighbors = [(sim_u, self.sim[u, sim_u], sim_r) for sim_u, sim_r in zip(ll, rr)]
        # 按相似度大小（元组的第1个）降序排序
        neighbors = sorted(neighbors, key=lambda tup: tup[1], reverse=True)[0:self.top_k]
        est = self.user_means[u]
        sum_u = 0
        divisor = 0
        
        for sim_u, sim, sim_r in neighbors:
            bias = sim_r - self.user_means[sim_u]
            sum_u += sim * bias
            divisor += sim
        
        if divisor != 0:
            est += sum_u / divisor
        return est


if __name__ == '__main__':
    file_name1 = "../data/ml-100k/ratings.dat"
    file_name2 = "../data/ml-100k/users.dat"
    data_builder = DataBuilderPlus(file_name1, file_name2, k_folds=5, shuffle=False, just_test_one=True)
    ratings = [(1, 1, 4, 0), (1, 2, 3, 0), (2, 1, 4, 0), (3, 1, 4, 0), (3, 2, 4, 0), (3, 3, 3, 0)]
    train_set = data_builder.mapping(ratings)
    logger.debug('train_set:')
    print(train_set.matrix.A)
    users = [('1', '18', 'M', 19, 85711), ('2', '50', 'F', 13, 94043),
             ('3', '18', 'M', 20, 32067)]
    logger.debug('user_set')
    print(np.array(users))
    user_cf = UserCfPre(min_coo=0)
    user_cf.train(train_set, users)
    print(user_cf.predict(2, 2))
