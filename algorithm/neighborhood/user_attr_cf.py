# -*- coding:utf-8 -*-

import numpy as np
from scipy.sparse import lil_matrix

from algorithm.mf.baseline import Baseline
from databuilder import DataBuilderPlus
from logger import logger
from neighborhood.user_cf import UserCF


class UserAttrCF(UserCF):
    def __init__(self, min=2, topk=20, use_baseline=False):
        super().__init__(min, topk, use_baseline)
        self.weight = np.array([0.4, 0.3, 0.2, 0.1])

    def compute_attr_similarity(self, user_num, user_data):
        logger.debug("user cf: compute attr similarity")
        user_data = np.array(user_data)
        sim = lil_matrix((user_num, user_num), dtype=np.double)  # 属性相似度
        cur = 0
        for k1 in range(user_num):
            cur += 1
            for k2 in range(k1 + 1, user_num):
                a = user_data[k1][1:] == user_data[k2][1:]
                sim[k1, k2] = (a * self.weight).sum()

            self.progress(cur, user_num)

        row, col = sim.nonzero()
        sim[col, row] = sim[row, col]
        return sim.A  # 转化为数组

    def _train(self):
        if self.use_baseline:
            self.baseline = Baseline()
            self.baseline.train(self.train_dataset)

        user_num = self.train_dataset.matrix.shape[0]
        item_num = self.train_dataset.matrix.shape[1]
        self.sim = self.compute_attr_similarity(user_num, self.user_dataset)
        self.item_means = self.train_dataset.get_item_means()
        self.user_means = self.train_dataset.get_user_means()


if __name__ == '__main__':
    file_name1 = "../../data/test_ratings.dat"
    file_name2 = "../../data/test_user.dat"
    data_builder2 = DataBuilderPlus(file_name1, file_name2, k_folds=5, shuffle=False, just_test_one=True)
    itcf = UserAttrCF(min=0)
    for user_set, train_set, test_set in data_builder2.cross_validation():
        itcf.train(train_set, user_set)
        break
    print(itcf.user_dataset)
    print(itcf.train_dataset.matrix.A)
    print(itcf.sim)
    print(itcf.predict(1, 1))
