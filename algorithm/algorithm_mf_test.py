import json

from logger import logger
from mf.baseline import Baseline
from mf.explicit_als import ExplicitALS
from mf.implicit_als import ImplicitALS
from mf.svd import SVD
from mf.svdpp import SVDPlusPlus
from util.databuilder import DataBuilder


def test_baseline(data_builder, factor_num):
    baseline = Baseline()
    baseline.n_factors = factor_num
    return data_builder.eval_factors(baseline)


def test_svd(data_builder, factor_num):
    svd = SVD()
    svd.n_factors = factor_num
    return data_builder.eval_factors(svd)


def test_svdpp(data_builder, factor_num):
    svdpp = SVDPlusPlus()
    svdpp.n_factors = factor_num
    return data_builder.eval_factors(svdpp)


def test_explicit_als(data_builder):
    data_builder.eval(ExplicitALS())


def test_implicit_als(data_builder):
    data_builder.eval(ImplicitALS())


if __name__ == '__main__':
    def test_100k():
        file_name1 = "../data/ml-100k/ratings.dat"
        file_name2 = "../data/ml-100k/users.dat"
        output1 = "../output/ml-100k/"
        data_builder = DataBuilder(file_name1, k_folds=5, shuffle=True,
                                   just_test_one=False, save_sim=True, output_path=output1)
        factors = [i for i in range(5, 50, 5)]
        result = {'baseline': [], 'svd': [], 'svdpp': []}
        for factor_num in factors:
            logger.debug('factor_num=' + str(factor_num))
            result['baseline'].append(test_baseline(data_builder, factor_num))
        show_result(result, output1)
        for factor_num in factors:
            result['svd'].append(test_svd(data_builder, factor_num))
        show_result(result, output1)
        for factor_num in factors:
            result['svdpp'].append(test_svdpp(data_builder, factor_num))
        show_result(result, output1)


    def test_1m():
        file_name1 = "../data/ml-1m/ratings.dat"
        file_name2 = "../data/ml-1m/users.dat"
        output1 = "../output/ml-1m/"
        data_builder = DataBuilder(file_name1, k_folds=5, shuffle=True,
                                   just_test_one=False, save_sim=True, output_path=output1)
        factors = [i for i in range(5, 50, 5)]
        result = {'baseline': [], 'svd': [], 'svdpp': []}
        for factor_num in factors:
            result['baseline'].append(test_baseline(data_builder, factor_num))
        show_result(result, output1)
        for factor_num in factors:
            result['svd'].append(test_svd(data_builder, factor_num))
        show_result(result, output1)
        for factor_num in factors:
            result['svdpp'].append(test_svdpp(data_builder, factor_num))
        show_result(result, output1)


    def show_result(result, path):
        print(result)
        with open('{}data.json'.format(path), 'w') as f:
            json.dump(result, f)


    test_100k()
    # test_1m()
