# -*- coding:utf-8 -*-

import numpy as np
from scipy.sparse import lil_matrix

from algorithm.mf.baseline import Baseline
from databuilder import DataBuilderPlus
from logger import logger
from neighborhood.user_cf import UserCF


class UserSigmoidCF(UserCF):
    def __init__(self, min=2, topk=20, use_baseline=False):
        super().__init__(min, topk, use_baseline)

    def compute_sigmoid_similarity(self, user_num, item_num, users_ratings):
        logger.debug("user cf: compute sigmoid similarity")
        sim = lil_matrix((user_num, user_num), dtype=np.double)
        sig = lil_matrix((user_num, user_num), dtype=np.double)  # sigmoid相似和
        coo = lil_matrix((user_num, user_num), dtype=np.int)  # 共现矩阵
        uoo = np.zeros(user_num, dtype=np.int)  # 用户评分数目
        boo = lil_matrix((user_num, user_num), dtype=np.int)  # 评分数并集数目
        level = lil_matrix((user_num, user_num), dtype=np.double)  # 置信度矩阵
        cur = 0
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

        crow, ccol = coo.nonzero()
        level[crow, ccol] = coo[crow, ccol] / (uoo[crow] + uoo[ccol] - coo[crow, ccol])
        level[ccol, crow] = level[crow, ccol]
        # 只需要考虑非0相似度和
        row, col = sig.nonzero()
        # 相似矩阵
        sim[row, col] = sig[row, col] / coo[row, col]
        sim[col, row] = sim[row, col]
        ssim = sim.A * level.A

        for i in range(user_num):
            for j in range(i + 1, user_num):
                boo[i, j] = uoo[i] + uoo[j] - coo[i, j]
        r, c = boo.nonzero()
        boo[c, r] = boo[r, c]

        return ssim, boo.A

    def _train(self):
        if self.use_baseline:
            self.baseline = Baseline()
            self.baseline.train(self.train_dataset)

        user_num = self.train_dataset.matrix.shape[0]
        item_num = self.train_dataset.matrix.shape[1]
        self.sim, self.coo = self.compute_sigmoid_similarity(user_num, item_num, self.train_dataset.get_items())
        self.item_means = self.train_dataset.get_item_means()
        self.user_means = self.train_dataset.get_user_means()


if __name__ == '__main__':
    file_name1 = "../../data/test_ratings.dat"
    file_name2 = "../../data/test_user.dat"
    data_builder2 = DataBuilderPlus(file_name1, file_name2, k_folds=5, shuffle=False, just_test_one=True)
    itcf = UserSigmoidCF(min=0)
    for user_set, train_set, test_set in data_builder2.cross_validation():
        itcf.train(train_set, user_set)
        break
    print(itcf.user_dataset)
    print(itcf.train_dataset.matrix.A)
    print(itcf.sim)
    print(itcf.predict(1, 1))
