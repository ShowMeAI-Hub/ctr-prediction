"""
Created on July 13, 2020
Updated on May 18, 2021

model: Deep & Cross Network for Ad Click Predictions

@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input, Layer, Dropout


class CrossNetwork(Layer):
    def __init__(self, layer_num, w_r=1e-6, b_r=1e-6):
        """
        CrossNetwork
        :@param layer_num: cross network深度
        :@param w_r: w正则化系数
        :@param b_r: b正则化系数
        """
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.w_r = w_r
        self.b_r = b_r

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.cross_weights = [
            self.add_weight(name='w_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.w_r),
                            trainable=True
                            )
            for i in range(self.layer_num)]
        self.cross_bias = [
            self.add_weight(name='b_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.b_r),
                            trainable=True
                            )
            for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        x_0 = tf.expand_dims(inputs, axis=2)  # (batch_size, dim, 1)
        x_l = x_0  # (None, dim, 1)
        for i in range(self.layer_num):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])  # (batch_size, dim, dim)
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l  # (batch_size, dim, 1)
        x_l = tf.squeeze(x_l, axis=2)  # (batch_size, dim)
        return x_l


class MyDNN(Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """
        DNN
        :@param hidden_units: 隐层神经元个数列表
        :@param activation: 激活函数
        :@param dropout: 随机失活概率
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x

class DCN(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-6, cross_w_r=1e-6, cross_b_r=1e-6):
        """
        Deep&Cross网络
        :@param feature_columns: 稀疏特征列表
        :@param hidden_units: 隐层神级元个数列表
        :@param activation: 激活函数
        :@param dnn_dropout: 随机失活概率
        :@param embed_reg: embedding正则化系数
        :@param cross_w_r: cross network的w正则化系数
        :@param cross_b_r: cross network的b正则化系数
        """
        super(DCN, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.layer_num = len(hidden_units)
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.cross_network = CrossNetwork(self.layer_num, cross_w_r, cross_b_r)
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.dense_final = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        sparse_inputs = inputs
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        x = sparse_embed

        # Cross Network
        cross_x = self.cross_network(x)
        
        # DNN
        dnn_x = self.dnn_network(x)
        
        # 拼接
        total_x = tf.concat([cross_x, dnn_x], axis=-1)
        outputs = tf.nn.sigmoid(self.dense_final(total_x))
        return outputs

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()