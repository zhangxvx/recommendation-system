import numpy as np

import util.measure as ms
import util.tools as tl
from util.logger import logger


class Estimator(object):
    """
    基础算法流程
    """

    def __init__(self):
        self.sim = None
        self.train_dataset = None
        self.user_dataset = None

    def train(self, train_dataset, user_dataset=None):
        self.train_dataset = train_dataset
        self.user_dataset = user_dataset

        with tl.Timer() as t:
            self._train()
        logger.info("{} algorithm train process cost {:.3f} sec".format(self.__class__.__name__, t.interval))

    def _train(self):
        raise NotImplementedError()

    def predict(self, u, i):
        raise NotImplementedError()

    def estimate(self, raw_test_dataset, measures):
        with tl.Timer() as t:
            error = self._estimate(raw_test_dataset, measures)

        logger.info("{} algorithm predict process cost {:.3f} sec".format(self.__class__.__name__, t.interval))
        return error

    def _estimate(self, raw_test_dataset, measures):
        users_mean = self.train_dataset.get_user_means()  # {下标:平均分}
        items_mean = self.train_dataset.get_item_means()

        all_raw_test = len(raw_test_dataset)
        errors = []
        cur = 0
        alg_count = 0

        for raw_u, raw_i, r, _ in raw_test_dataset:
            cur += 1
            has_raw_u = raw_u in self.train_dataset.uid_dict
            has_raw_i = raw_i in self.train_dataset.iid_dict

            if not has_raw_u and not has_raw_i:
                real, est = r, self.train_dataset.global_mean
            elif not has_raw_u:
                i = self.train_dataset.iid_dict[raw_i]
                real, est = r, items_mean[i]
            elif not has_raw_i:
                u = self.train_dataset.uid_dict[raw_u]
                real, est = r, users_mean[u]
            else:
                u = self.train_dataset.uid_dict[raw_u]
                i = self.train_dataset.iid_dict[raw_i]
                real, est = r, self.predict(u, i)
                alg_count += 1

            est = min(5, est)
            est = max(1, est)
            errors.append(real - est)
            # print(real, est)

            self.progress(cur, all_raw_test)  # 进度条

        # 调用评价指标计算函数 getattr(ms, 'mae')(errors)相当于ms.mae(errors)
        fold_eval_result = [getattr(ms, measure)(errors) for measure in measures]
        return fold_eval_result

    @staticmethod
    def progress(cur, all_num):
        bin_num = int(all_num / 10)
        if bin_num == 0:
            bin_num = 1
        if cur % bin_num == 0 or cur >= all_num:
            progress = 100 * (cur / all_num)
            logger.debug("  progress: {:.2f}%".format(progress))


class IterationEstimator(Estimator):
    """适合迭代式算法"""

    def __init__(self):
        super().__init__()
        self.n_epochs = None

    def _train(self):
        self._prepare()
        for current_epoch in range(self.n_epochs):
            logger.debug(" processing epoch {}".format(current_epoch))
            self._iteration()
            logger.debug(" cur train rmse {}".format(self._eval()))

    def predict(self, u, i):
        raise NotImplementedError()

    def _prepare(self):
        """ 准备工作 """

        raise NotImplementedError()

    def _iteration(self):
        """ 核心迭代 """

        raise NotImplementedError()

    def _pred(self):
        """ 核心预测 """
        raise NotImplementedError()

    def _eval(self):
        """ 整体指标评估 """

        pred_ratings = self._pred()
        real_ratings = self.train_dataset.matrix
        idx = real_ratings.nonzero()
        bias = np.asarray(pred_ratings[idx] - real_ratings[idx])
        return np.sqrt(np.sum(bias ** 2) / real_ratings.count_nonzero())
