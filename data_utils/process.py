"""
criteo数据集预处理

criteo数据特征说明：
- Label - 标签列，「点击」取值为1，「未点击」取值为0
- I1-I13 - 总共13列整型特征（绝大多数是计数特征）
- C1-C26 - 总共26列类别型特征，基于脱敏原因，数据经过哈希处理得到32位串
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split

def sparse_feature(feat, feat_num, embed_dim=4):
    """
    为稀疏特征构建字典
    :@param feat: 特征名称
    :@param feat_num: 不重复的稀疏特征个数
    :@param embed_dim: 特征嵌入(embedding)的维度
    """
    return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def dense_feature(feat):
    """
    为稠密(数值)型特征构建字典
    :@param feat: 特征名称
    """
    return {'feat_name': feat}


def create_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
    """
    criteo数据集预处理
    :@param file: 数据路径
    :@param embed_dim: 稀疏特征的嵌入(embedding)维度
    :@param read_part: 读取部分数据(在数据集很大的情况下最好设定为True)
    :@param sample_num: 部分读取的形态下，每个part的样本量
    :@param test_size: 测试集比例
    """
    names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13',
             'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
             'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
             'C23', 'C24', 'C25', 'C26']

    # 部分读取与全部读取
    if read_part:
        data_df = pd.read_csv(file, sep='\t', iterator=True, header=None, names=names)
        data_df = data_df.get_chunk(sample_num)

    else:
        data_df = pd.read_csv(file, sep='\t', header=None, names=names)

    # 指定稀疏特征与稠密特征
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    # 缺失值填充
    data_df[sparse_features] = data_df[sparse_features].fillna('nan')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    # 离散化处理
    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    data_df[dense_features] = est.fit_transform(data_df[dense_features])

    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # 特征工程：对离散特征进行embedding处理
    feature_columns = [sparse_feature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim) for feat in features]
    train, test = train_test_split(data_df, test_size=test_size)

    # 生成训练与测试集
    train_X = train[features].values.astype('int32')
    train_y = train['label'].values.astype('int32')
    test_X = test[features].values.astype('int32')
    test_y = test['label'].values.astype('int32')

    return feature_columns, (train_X, train_y), (test_X, test_y)