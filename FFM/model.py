"""
FFM模型TensorFlow2.X构建
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.regularizers import l2


class MyLayer(Layer):
    def __init__(self, sparse_feature_columns, k, w_r=1e-6, v_r=1e-6):
        """
        :@param feature_columns: A list. sparse column feature information.
        :@param k: 隐向量维度
        :@param w_r: 参数w的正则化系数
        :@param v_r: 参数v的正则化系数
        """
        super(FFM_Layer, self).__init__()
        self.sparse_feature_columns = sparse_feature_columns
        self.k = k
        self.w_r = w_r
        self.v_r = v_r
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.field_num = len(self.sparse_feature_columns)

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_r),
                                 trainable=True)
        self.v = self.add_weight(name='v',
                                 shape=(self.feature_length, self.field_num, self.k),
                                 initializer='random_normal',
                                 regularizer=l2(self.v_r),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        
        # 一阶项
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        
        # 二阶项
        second_order = 0
        latent_vector = tf.reduce_sum(tf.nn.embedding_lookup(self.v, inputs), axis=1)  # (batch_size, field_num, k)
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                second_order += tf.reduce_sum(latent_vector[:, i] * latent_vector[:, j], axis=1, keepdims=True)
        
        return first_order + second_order

class FFM(Model):
    def __init__(self, feature_columns, k, w_r=1e-6, v_r=1e-6):
        """
        FFM模型
        :@param feature_columns: A list. sparse column feature information.
        :@param k: 隐向量维度
        :@param w_r: 参数w的正则化系数
        :@param v_r: 参数v的正则化系数
        """
        super(FFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.ffm = MyLayer(self.sparse_feature_columns, k, w_r, v_r)

    def call(self, inputs, **kwargs):
        ffm_out = self.ffm(inputs)
        outputs = tf.nn.sigmoid(ffm_out)
        return outputs

    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        tf.keras.Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()