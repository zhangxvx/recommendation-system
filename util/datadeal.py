# -*- coding: utf-8 -*-
import itertools

import numpy as np


def deal_ml_100k(path):
    deal_users(path)


def deal_ratings(path):
    with open('{}/u.data'.format(path), 'r', encoding='utf-8') as f:
        raws = [parse_line_rating(line) for line in itertools.islice(f, 0, None)]
    
    with open('{}/ratings.dat'.format(path), 'w', encoding='utf-8') as f:
        for raw in raws:
            f.writelines('::'.join(raw) + '\n')


def deal_users(path):
    with open('{}/u.user'.format(path), 'r', encoding='utf-8') as f:
        raws = np.array([parse_line_user(line) for line in itertools.islice(f, 0, None)])
    deal_age(raws[:, 1])
    deal_work(raws[:, 3])
    with open('{}/user_data.dat'.format(path), 'w', encoding='utf-8') as f:
        for raw in raws:
            f.writelines('::'.join(raw) + '\n')


def deal_age(age):
    """
    - Age is chosen from the following ranges:
        *  1:  "Under 18"
        * 18:  "18-24"
        * 25:  "25-34"
        * 35:  "35-44"
        * 45:  "45-49"
        * 50:  "50-55"
        * 56:  "56+"
    """
    age[age < '18'] = '1'
    age[('17' < age) & (age < '25')] = '18'
    age[('24' < age) & (age < '35')] = '25'
    age[('34' < age) & (age < '45')] = '35'
    age[('44' < age) & (age < '50')] = '45'
    age[('49' < age) & (age < '56')] = '50'
    age['55' < age] = '56'


def deal_work(work):
    """
   - Occupation is chosen from the following choices:
       0：administrator
       1：artist
       2：doctor
       3：educator
       4：engineer
       5：entertainment
       6：executive
       7：healthcare
       8：homemaker
       9:lawyer
       10:librarian
       11:marketing
       12:none
       13:other
       14:programmer
       15:retired
       16:salesman
       17:scientist
       18:student
       19:technician
       20:writer
   """
    work[work == 'administrator'] = '0'
    work[work == 'artist'] = '1'
    work[work == 'doctor'] = '2'
    work[work == 'educator'] = '3'
    work[work == 'engineer'] = '4'
    work[work == 'entertainment'] = '5'
    work[work == 'executive'] = '6'
    work[work == 'healthcare'] = '7'
    work[work == 'homemaker'] = '8'
    work[work == 'lawyer'] = '9'
    work[work == 'librarian'] = '10'
    work[work == 'marketing'] = '11'
    work[work == 'none'] = '12'
    work[work == 'other'] = '13'
    work[work == 'programmer'] = '14'
    work[work == 'retired'] = '15'
    work[work == 'salesman'] = '16'
    work[work == 'scientist'] = '17'
    work[work == 'student'] = '18'
    work[work == 'technician'] = '19'
    work[work == 'writer'] = '20'


def parse_line_rating(line):
    """分割行——评分数据"""
    line = line.split("\t")
    uid, iid, r, timestamp = (line[i].strip() for i in range(4))
    return uid, iid, r, timestamp


def parse_line_user(line):
    """分割行——用户数据"""
    line = line.split("|")
    uid, age, sex, work, zip_code = (line[i].strip() for i in range(5))
    return uid, age, sex, work, zip_code


if __name__ == '__main__':
    path = '../data/ml-100k'
    deal_ml_100k(path)
