# -*- coding:utf-8 -*-

import itertools
import os

import numpy as np
from scipy.sparse import csr_matrix

from util.logger import logger
from util.matrix import Matrix
from util.tools import print_pretty


class DataBuilder(object):
    """
    构造数据模型
       
    参数
    ----------    
    rate_data : 文件地址，这里用的grouplens数据集
    k_folds : k折交叉验证
    shuffle : 是否对数据shuffle(随机，洗牌)
    just_test_one : k折交叉验证要运行k次，这里只运行一次，方便测试程序正确性
    """

    def __init__(self, rate_data, k_folds=5, shuffle=False, just_test_one=True, save_sim=False, output_path=None):
        self.rate_data = rate_data
        self.k_folds = k_folds
        self.shuffle = shuffle
        self.just_test_one = just_test_one
        self.save_sim = save_sim
        self.output_path = output_path

    def cross_validation(self):
        """交叉验证"""
        return self._cv()

    def _cv(self):
        raw_ratings = self.read_ratings()

        if self.shuffle:
            np.random.shuffle(raw_ratings)

        stop = 0
        raw_len = len(raw_ratings)
        offset = raw_len // self.k_folds
        left = raw_len % self.k_folds
        for fold_i in range(self.k_folds):
            logger.debug("current fold {}".format(fold_i + 1))
            start = stop
            stop += offset
            if fold_i < left:
                stop += 1  # 前left组多加一个，保持k_folds组

            # 使用生成器，提高效率
            yield self.mapping(raw_ratings[:start] + raw_ratings[stop:]), raw_ratings[start:stop]

    def eval(self, algorithm, measures=None):
        """评估"""
        if measures is None:
            measures = ["rmse", "mae"]
        logger.info('algorithm:' + algorithm.__class__.__name__)
        eval_results = []
        for train_set, test_set in self.cross_validation():
            algorithm.train(train_set)
            eval_results.append(algorithm.estimate(test_set, measures))
            if self.just_test_one:
                break

        print_pretty(algorithm.__class__.__name__, measures, eval_results)

    def read_ratings(self):
        """ 读取数据 """
        with open(self.rate_data) as f:
            raw = [self.parse_line_rating(line) for line in itertools.islice(f, 0, None)]
        return raw

    def eval_k(self, algorithm, topks=None, measures=None):
        """评估"""
        if measures is None:
            measures = ["rmse", "mae"]
        logger.info('algorithm:' + algorithm.__class__.__name__)

        eval_results = []
        cur = 0
        for train_set, test_set in self.cross_validation():
            cur += 1
            algorithm.train(train_set)

            if self.save_sim:
                temp_path = '{}/sim/{}/'.format(self.output_path, algorithm.__class__.__name__)
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)

                np.savetxt('{}/fold{}.csv'.format(temp_path, cur), algorithm.sim, fmt='%0.3f')

            eval_result_fold = []
            for k in topks:
                logger.info('k={}'.format(k))
                algorithm.topk = k
                eval_result_fold.append(algorithm.estimate(test_set, measures))
            eval_results.append(eval_result_fold)

            if self.just_test_one:
                break
        self.show_result_k(algorithm, measures, eval_results, topks)

    def show_result_k(self, algorithm, measures, eval_results, topks):
        pad = '{:<9}' * (len(measures) + 1)
        keep = lambda res: ['{:0.3f}'.format(eval) for eval in res]
        with open('{}/{}.txt'.format(self.output_path, algorithm.__class__.__name__), 'w', encoding='utf-8') as file:
            print(eval_results)
            for i in range(len(eval_results)):
                file.write('[')
                for j in range(len(eval_results[i])):
                    file.write('[')
                    file.write(','.join(keep(eval_results[i][j])))
                    file.write(']')
                    if j != len(eval_results[i]) - 1:
                        file.write(',')
                if i != len(eval_results) - 1:
                    file.write('],\n')
                else:
                    file.write(']\n\n')
            avg = np.mean(eval_results, axis=0).T
            print(avg)
            for i in range(len(avg)):
                file.write('[')
                file.write(','.join(keep(avg[i])))
                if i != len(avg) - 1:
                    file.write('],\n')
                else:
                    file.write(']\n\n')

            print(pad.format('', *measures))
            file.write(pad.format('', *measures))
            file.write('\n')
            eval_results = np.array(eval_results)
            for i in range(len(topks)):
                res_k = np.mean(eval_results[:, i], axis=0)
                print(pad.format('k= {}'.format(topks[i]), *keep(res_k)))
                file.write(pad.format('k= {}'.format(topks[i]), *keep(res_k)))
                file.write('\n')

    def eval_factors(self, algorithm, factors=None, measures=None):
        if measures is None:
            measures = ["rmse", "mae"]
        logger.info('algorithm:' + algorithm.__class__.__name__)
        eval_results = []
        for train_set, test_set in self.cross_validation():
            algorithm.train(train_set)
            eval_results.append(algorithm.estimate(test_set, measures))
            if self.just_test_one:
                break
        return eval_results

    @staticmethod
    def parse_line_rating(line):
        """分割行——评分数据"""
        line = line.split("::")
        uid, iid, r, timestamp = (line[i].strip() for i in range(4))
        return int(uid), int(iid), float(r), timestamp

    @staticmethod
    def mapping(train_ratings):
        uid_dict = {}  # {"编号":"下标（编号-1）"}
        iid_dict = {}
        row = []  # user 下标
        col = []  # item 下标
        data = []

        for uid, iid, r, timestamp in train_ratings:
            try:
                uid_index = uid_dict[uid]
            except KeyError:
                uid_index = uid - 1
                uid_dict[uid] = uid_index
            try:
                iid_index = iid_dict[iid]
            except KeyError:
                iid_index = iid - 1
                iid_dict[iid] = iid_index
            row.append(uid_index)
            col.append(iid_index)
            data.append(r)
        sparse_matrix = csr_matrix((data, (row, col)))

        return Matrix(sparse_matrix, uid_dict, iid_dict)


class DataBuilderPlus(DataBuilder):
    """
    构造数据模型

    参数
    ----------
    user_data: 用户信息数据集
    """

    def __init__(self, rate_data, user_data, k_folds=5, shuffle=False, just_test_one=True, save_sim=False,
                 output_path=None):
        super().__init__(rate_data, k_folds, shuffle, just_test_one, save_sim, output_path)
        self.user_data = user_data

    def read_users(self):
        with open(self.user_data) as f:
            raw = [self.parse_line_user(line) for line in itertools.islice(f, 0, None)]
        return raw

    def _cv(self):
        raw_ratings = self.read_ratings()
        raw_users = self.read_users()

        if self.shuffle:
            np.random.shuffle(raw_ratings)

        stop = 0
        raw_len = len(raw_ratings)
        offset = raw_len // self.k_folds
        left = raw_len % self.k_folds
        for fold_i in range(self.k_folds):
            logger.info("current fold {}".format(fold_i + 1))
            start = stop
            stop += offset
            if fold_i < left:
                stop += 1  # 前left组多加一个，保持k_folds组

            # 使用生成器，提高效率
            yield raw_users, self.mapping(raw_ratings[:start] + raw_ratings[stop:]), raw_ratings[start:stop]

    def eval(self, algorithm, measures=None):
        """评估"""
        if measures is None:
            measures = ["rmse", "mae"]
        logger.info('algorithm:' + algorithm.__class__.__name__)
        eval_results = []
        for user_set, train_set, test_set in self.cross_validation():
            algorithm.train(train_set, user_set)
            eval_results.append(algorithm.estimate(test_set, measures))
            if self.just_test_one:
                break

        print_pretty(algorithm.__class__.__name__, measures, eval_results, self.output_path)

    def eval_multi(self, algorithm1, algorithm2, measures=None):
        """评估"""
        if measures is None:
            measures = ["rmse", "mae"]
        logger.info('algorithm:' + algorithm1.__class__.__name__ + ' ' + algorithm2.__class__.__name__)
        eval_results = []
        for user_set, train_set, test_set in self.cross_validation():
            algorithm1.train(train_set, user_set)
            algorithm1.fill()
            algorithm2.train_set_filled = algorithm1.train_set_filled
            algorithm2.train(train_set, user_set)
            eval_results.append(algorithm2.estimate(test_set, measures))
            if self.just_test_one:
                break

        self.show_result(measures, eval_results)

    def show_result(self, measures, eval_results):
        pad = '{:<9}' * (len(measures) + 1)
        keep = lambda res: ['{:0.3f}'.format(eval) for eval in res]
        with open('{}/{}.txt'.format(self.output_path, 'multi'), 'w', encoding='utf-8') as file:
            print(pad.format('', *measures))
            file.write(pad.format('', *measures))
            file.write('\n')
            for i, eval_result in enumerate(eval_results):
                print(pad.format('fold {}'.format(i), *keep(eval_result)))
                file.write(pad.format('fold {}'.format(i), *keep(eval_result))),
                file.write('\n')
            print(pad.format('avg', *keep(np.mean(eval_results, axis=0))))
            file.write(pad.format('avg', *keep(np.mean(eval_results, axis=0))))
            file.write('\n')

    @staticmethod
    def parse_line_user(line):
        """分割行——用户数据"""
        line = line.split("::")
        uid, age, sex, work, zip_code = (line[i].strip() for i in range(5))
        return int(uid), age, sex, work, zip_code


if __name__ == '__main__':
    file_name1 = "../data/ml-100k/ratings.dat"
    file_name2 = "../data/ml-100k/users.dat"

    data_builder = DataBuilder(file_name1, k_folds=5, shuffle=False, just_test_one=True)
    for train_dataset, test_dataset in data_builder.cross_validation():
        logger.info(train_dataset.matrix.shape)
        break

    data_builder2 = DataBuilderPlus(file_name1, file_name2, k_folds=5, shuffle=False, just_test_one=True)
    for user_dataset, train_dataset, test_dataset in data_builder2.cross_validation():
        logger.info('(%s, %s)' % (len(user_dataset), len(user_dataset[0])))
        logger.info(train_dataset.matrix.shape)
        break
