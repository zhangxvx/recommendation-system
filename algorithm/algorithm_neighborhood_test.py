from neighborhood.item_cf import ItemCF
from neighborhood.item_sigmoid_cf import ItemSigmoidCF
from neighborhood.user_attr_cf import UserAttrCF
from neighborhood.user_cf import UserCF
from neighborhood.user_mixed_cos_cf import UserMixedCosCF
from neighborhood.user_mixed_sigmoid_cf import UserMixedSigmoidCF
from neighborhood.user_sigmoid_cf import UserSigmoidCF
from util.databuilder import DataBuilderPlus, DataBuilder


def test_itemcf(data_builder):
    cf = ItemCF()
    data_builder.eval_k(cf, topks)


def test_itemsigmoidcf(data_builder):
    cf = ItemSigmoidCF()
    data_builder.eval_k(cf, topks)


def test_usercf(data_builder):
    cf = UserCF()
    data_builder.eval_k(cf, topks)


def test_userattrcf(data_builder):
    cf = UserAttrCF()
    data_builder.eval_k(cf, topks)


def test_usersigmoidcf(data_builder):
    cf = UserSigmoidCF()
    data_builder.eval_k(cf, topks)


def test_usermixedcoscf(data_builder):
    cf = UserMixedCosCF()
    data_builder.eval_k(cf, topks)


def test_usermixedsigmoidcf(data_builder):
    cf = UserMixedSigmoidCF()
    data_builder.eval_k(cf, topks)


if __name__ == '__main__':
    topks = [5, 10, 15, 20, 30, 40, 50, 60, 80]


    def test_100k():
        file_name1 = "../data/ml-100k/ratings.dat"
        file_name2 = "../data/ml-100k/users.dat"
        output1 = "../output/ml-100k/"
        # data_builder = DataBuilderPlus(file_name1, user_data=file_name2, k_folds=5, shuffle=True,
        # just_test_one=False, save_sim=True, output_path=output1)
        data_builder = DataBuilder(file_name1, k_folds=5, shuffle=True,
                                   just_test_one=False, save_sim=True, output_path=output1)
        # test_userattrcf(data_builder)
        test_usercf(data_builder)
        test_usersigmoidcf(data_builder)
        # test_usermixedcoscf(data_builder)
        # test_usermixedsigmoidcf(data_builder)
        test_itemcf(data_builder)
        test_itemsigmoidcf(data_builder)


    def test_1m():
        file_name1 = "../data/ml-1m/ratings.dat"
        file_name2 = "../data/ml-1m/users.dat"
        output1 = "../output/ml-1m/"
        # data_builder = DataBuilderPlus(file_name1, user_data=file_name2, k_folds=5, shuffle=True,
        #                                just_test_one=False, save_sim=True, output_path=output1)
        data_builder = DataBuilder(file_name1, k_folds=5, shuffle=True,
                                   just_test_one=False, save_sim=True, output_path=output1)
        test_usercf(data_builder)
        test_usersigmoidcf(data_builder)
        test_itemcf(data_builder)
        test_itemsigmoidcf(data_builder)


    # test_100k()
    test_1m()
