# -*- coding:utf-8 -*-

import numpy as np

from logger import logger
from mf.baseline import Baseline
from neighborhood.user_attr_cf import UserAttrCF
from neighborhood.user_sigmoid_cf import UserSigmoidCF
from util.databuilder import DataBuilderPlus


class UserMixedCosCF(UserAttrCF, UserSigmoidCF):
    def __init__(self, min=2, topk=20, use_baseline=False):
        super().__init__(min, topk, use_baseline)

    def compute_mixed_similarity(self):
        logger.debug("user cf: compute mixed similarity")
        exp = np.exp(-self.boo)
        beta = 2 * exp / (1 + exp)
        sim = beta * self.attr_sim + (1 - beta) * self.act_sim
        row, col = sim.nonzero()
        sim[col, row] = sim[row, col]
        return sim

    def _train(self):
        if self.use_baseline:
            self.baseline = Baseline()
            self.baseline.train(self.train_dataset)

        user_num = self.train_dataset.matrix.shape[0]
        item_num = self.train_dataset.matrix.shape[1]

        self.attr_sim = self.compute_attr_similarity(user_num, self.user_dataset)
        self.act_sim, self.boo = self.compute_cosine_similarity(user_num, item_num, self.train_dataset.get_items())
        self.sim = self.compute_mixed_similarity()

        self.item_means = self.train_dataset.get_item_means()
        self.user_means = self.train_dataset.get_user_means()


if __name__ == '__main__':
    file_name1 = "../../data/test_ratings.dat"
    file_name2 = "../../data/test_user.dat"
    data_builder2 = DataBuilderPlus(file_name1, file_name2, k_folds=5, shuffle=False, just_test_one=True)
    itcf = UserMixedCosCF(min=0)
    for user_set, train_set, test_set in data_builder2.cross_validation():
        itcf.train(train_set, user_set)
        break
    print(itcf.user_dataset)
    print(itcf.train_dataset.matrix.A)
    print(itcf.attr_sim)
    print(itcf.act_sim)
    print(itcf.sim)
    print(itcf.predict(1, 1))
