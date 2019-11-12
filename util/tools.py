import time
from functools import wraps

import numpy as np


def print_pretty(name, measures, eval_results, path='output/'):
    """ 格式化输出 """
    pad = '{:<9}' * (len(measures) + 1)
    txt = '{}{}'.format(pad, '\n')
    keep = lambda eval_result: ['{:.4f}'.format(single_eval) for single_eval in eval_result]
    with open('{}{}.txt'.format(path, name), 'w', encoding='utf-8') as file:
        print(pad.format('', *measures))
        file.write(txt.format('', *measures, '\n'))
        for i, eval_result in enumerate(eval_results):
            print(pad.format('fold {}'.format(i), *keep(eval_result)))
            file.write(txt.format('fold {}'.format(i), *keep(eval_result), '\n'))
        print(pad.format('avg', *keep(np.mean(eval_results, axis=0))))
        file.write(txt.format('avg', *keep(np.mean(eval_results, axis=0)), '\n'))


class Timer(object):
    """
    time util
    """

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


def timer(func):
    """
    修饰器计时器
    :param func: 被修饰函数
    :return: 修饰后的函数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        exe_time = end_time - start_time
        print("%s process cost %0.3fs" % (func.__name__, exe_time))

    return wrapper


@timer
def test_timer():
    print([i for i in range(100000)])


if __name__ == '__main__':
    test_timer()
