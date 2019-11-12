# -*- coding:utf-8 -*-

import numpy as np
from scipy.sparse import lil_matrix

from algorithm.estimator import Estimator
from algorithm.mf.baseline import Baseline
from databuilder import DataBuilder, DataBuilderPlus
from logger import logger


class UserCF(Estimator):
    """
    属性
    ---------
    min_coo : 有效交互数下限
    top_k : 相似矩阵topk
    use_baseline : 是否嵌入baseline计算bias
    """

    def __init__(self, min=2, topk=20, use_baseline=False):
        super().__init__()
        self.min = min
        self.topk = topk
        self.use_baseline = use_baseline

    def compute_cosine_similarity(self, user_num, item_num, users_ratings):
        logger.debug("user cf: compute cosine similarity")
        sim = lil_matrix((user_num, user_num), dtype=np.double)
        dot = lil_matrix((user_num, user_num), dtype=np.double)  # 点积
        sq = np.zeros(user_num, dtype=np.double)  # 向量平方和
        coo = lil_matrix((user_num, user_num), dtype=np.int)  # 共现矩阵
        uoo = np.zeros(user_num, dtype=np.int)  # 用户评分数目
        boo = lil_matrix((user_num, user_num), dtype=np.int)  # 评分数并集数目
        cur = 0
        for i, (u, r) in users_ratings:
            cur = cur + 1
            for k1 in range(len(u)):
                uoo[u[k1]] += 1
                sq[u[k1]] += r[k1] ** 2
                for k2 in range(k1 + 1, len(u)):
                    u1, u2 = u[k1], u[k2]
                    if u1 > u2:
                        u1, u2 = u2, u1
                        k1, k2 = k2, k1
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

        for i in range(user_num):
            for j in range(i + 1, user_num):
                boo[i, j] = uoo[i] + uoo[j] - coo[i, j]

        r, c = boo.nonzero()
        boo[c, r] = boo[r, c]
        return sim.A, boo.A

    def _train(self):
        if self.use_baseline:
            self.baseline = Baseline()
            self.baseline.train(self.train_dataset)

        user_num = self.train_dataset.matrix.shape[0]
        item_num = self.train_dataset.matrix.shape[1]
        self.sim, self.coo = self.compute_cosine_similarity(user_num, item_num, self.train_dataset.get_items())
        self.item_means = self.train_dataset.get_item_means()
        self.user_means = self.train_dataset.get_user_means()

    def predict(self, u, i):
        ll, rr = self.train_dataset.get_item(i)  # ([行下标],[评分])
        neighbors = [(sim_u, self.sim[u, sim_u], sim_r) for sim_u, sim_r in zip(ll, rr)]
        # 按相似度大小（元组的第1个）降序排序
        neighbors = sorted(neighbors, key=lambda tup: tup[1], reverse=True)[0:self.topk]
        est = self.baseline.predict(u, i) if self.use_baseline else self.user_means[u]
        sum_u = 0
        divisor = 0
        for sim_u, sim, sim_r in neighbors:
            # if not self.use_baseline:
            #     bias = sim_r - self.user_means[sim_u]
            # else:
            #     bias = sim_r - self.baseline.predict(sim_u, i)
            # sum_u += sim * bias
            sum_u += sim * sim_r
            divisor += sim
        if divisor != 0:
            # est += sum_u / divisor
            est = sum_u / divisor
        return est


if __name__ == '__main__':
    file_name1 = "../../data/test_ratings.dat"
    file_name2 = "../../data/test_user.dat"
    data_builder2 = DataBuilderPlus(file_name1, file_name2, k_folds=5, shuffle=False, just_test_one=True)
    itcf = UserCF(min=0)
    for user_set, train_set, test_set in data_builder2.cross_validation():
        itcf.train(train_set, user_set)
        break
    print(itcf.user_dataset)
    print(itcf.train_dataset.matrix.A)
    print(itcf.sim)
    print(itcf.predict(1, 1))
