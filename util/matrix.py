# -*- coding:utf-8 -*-

import numpy as np


class Matrix(object):
    
    def __init__(self, sparse_matrix, uid_dict=None, iid_dict=None):
        self.matrix = sparse_matrix.tocsc()
        self.uids = self.get_uids()
        self.iids = self.get_iids()
        self.uid_dict = uid_dict
        self.iid_dict = iid_dict
        self.global_mean = np.mean(sparse_matrix.data)
    
    def get_item(self, i):
        """ (is, (us, rs)) """
        ratings = self.matrix.getcol(i).tocoo()
        return ratings.row, ratings.data
    
    def get_user(self, u):
        """ (u, (is, rs))  """
        ratings = self.matrix.getrow(u).tocoo()
        return ratings.col, ratings.data
    
    def get_items(self):
        """ iterator(i, (us, rs)) """
        for i in self.get_iids():
            yield i, self.get_item(i)
    
    def get_users(self):
        """ iterator(u, (is, rs)) """
        for u in self.get_uids():
            yield u, self.get_user(u)
    
    def get_uids(self):
        """ 所有用户id集  """
        return set(self.matrix.tocoo().row)
    
    def get_iids(self):
        """ 所有物品id集  """
        return set(self.matrix.tocoo().col)
    
    def get_user_means(self):
        """ 用户的平均评分字典 """
        users_mean = {}
        for u in self.get_uids():
            users_mean[u] = np.mean(self.get_user(u)[1])
        return users_mean
    
    def get_item_means(self):
        """ 物品的平均评分字典  """
        item_means = {}
        for i in self.get_iids():
            item_means[i] = np.mean(self.get_item(i)[1])
        return item_means
    
    def all_ratings(self):
        """ iterator(u,i,r) """
        coo_matrix = self.matrix.tocoo()
        return zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
    
    def has_user(self, u):
        """ 是否存在用户u  """
        return u in self.uids
    
    def has_item(self, i):
        """ 是否存在物品i  """
        return i in self.iids
    
    def get_uid_dict(self):
        uid_dict = {}
        for i in self.get_uids():
            uid_dict[i + 1] = i
        return uid_dict
    
    def get_iid_dict(self):
        iid_dict = {}
        for i in self.get_iids():
            iid_dict[i + 1] = i
        return iid_dict


if __name__ == '__main__':
    from scipy.sparse import csr_matrix
    
    mat = Matrix(csr_matrix(([1, 2, 3, 4, 5, 6], ([0, 0, 0, 1, 1, 3], [0, 1, 2, 0, 1, 2]))))
    print(mat.matrix.A)
    print(mat.iid_dict)
    print(mat.iids)
    print(mat.global_mean)
