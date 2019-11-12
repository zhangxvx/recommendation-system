# -*- coding:utf-8 -*-

import numpy as np
from scipy.sparse import lil_matrix

from algorithm.estimator import Estimator
from algorithm.mf.baseline import Baseline
from databuilder import DataBuilder, DataBuilderPlus
from logger import logger


class ItemCF(Estimator):
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
        logger.debug("item cf: compute cosine similarity")
        sim = lil_matrix((item_num, item_num), dtype=np.double)
        dot = lil_matrix((item_num, item_num), dtype=np.double)  # 点积
        sq = np.zeros(item_num, dtype=np.double)  # item向量平方和
        coo = lil_matrix((item_num, item_num), dtype=np.double)  # 共现矩阵
        ioo = np.zeros(item_num, dtype=np.int)  # 用户评分数目
        boo = lil_matrix((item_num, item_num), dtype=np.int)  # 评分数并集数目
        cur = 0
        for u, (ii, rr) in users_ratings:
            cur = cur + 1
            for k1 in range(len(ii)):
                ioo[ii[k1]] += 1
                sq[ii[k1]] += rr[k1] ** 2
                for k2 in range(k1 + 1, len(ii)):
                    i1, i2 = ii[k1], ii[k2]
                    if i1 > i2:
                        i1, i2 = i2, i1
                        k1, k2 = k2, k1
                    dot[i1, i2] += rr[k1] * rr[k2]
                    coo[i1, i2] += 1
            self.progress(cur, user_num)

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
                boo[i, j] = ioo[i] + ioo[j] - coo[i, j]

        r, c = boo.nonzero()
        boo[c, r] = boo[r, c]
        # print(sim.A)
        # print(boo.A)
        # print(coo.A)
        return sim.A, boo.A

    def _train(self):
        if self.use_baseline:
            self.baseline = Baseline()
            self.baseline.train(self.train_dataset)

        user_num = self.train_dataset.matrix.shape[0]
        item_num = self.train_dataset.matrix.shape[1]
        self.sim, self.boo = self.compute_cosine_similarity(user_num, item_num, self.train_dataset.get_users())

        self.item_means = self.train_dataset.get_item_means()
        self.user_means = self.train_dataset.get_user_means()

    def predict(self, u, i):
        ll, rr = self.train_dataset.get_user(u)  # ([列下标],[评分])
        neighbors = [(sim_i, self.sim[i, sim_i], sim_r) for sim_i, sim_r in zip(ll, rr)]
        # 按相似度大小（元组的第1个）降序排序
        neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)[0:self.topk]
        est = self.baseline.predict(u, i) if self.use_baseline else self.item_means[i]
        sum_i = 0
        divisor = 0
        for sim_i, sim, sim_r in neighbors:
            # if not self.use_baseline:
            #     bias = sim_r - self.item_means[sim_i]
            # else:
            #     bias = sim_r - self.baseline.predict(u, sim_i)
            #
            # sum_i += sim * bias
            sum_i += sim * sim_r
            # divisor += sim
            divisor += sim

        if divisor != 0:
            # est += sum_i / divisor
            est = sum_i / divisor
        return est


if __name__ == '__main__':
    file_name1 = "../../data/test_ratings.dat"
    file_name2 = "../../data/test_user.dat"
    data_builder = DataBuilderPlus(file_name1, file_name2, 5, False, True, False, '../../data/')
    itcf = ItemCF(min=0)
    # data_builder.eval_k(itcf)
    for user_set, train_set, test_set in data_builder.cross_validation():
        itcf.train(train_set)
        break
    print(itcf.train_dataset.matrix.A)
    print(itcf.sim)
    print(itcf.predict(1, 1))
