"""
Wide&Deep模型TensorFlow2.X构建
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input, Layer
from tensorflow.keras.regularizers import l2


class MyLinear(Layer):
    def __init__(self, feature_length, w_r=1e-6):
        """
        Linear部分
        :@param feature_length: 特征长度
        :@param w_r: 参数w的正则化系数
        """
        super(Linear, self).__init__()
        self.feature_length = feature_length
        self.w_r = w_r

    def build(self, input_shape):
        self.w = self.add_weight(name="w",
                                 shape=(self.feature_length, 1),
                                 regularizer=l2(self.w_r),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        result = tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        return result


class MyDNN(Layer):
    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """
        DNN部分
        :@param hidden_units: 隐层节点个数列表
        :@param activation: 激活函数
        :@param dropout: 随机失活的概率
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


class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0., embed_reg=1e-6, w_r=1e-6):
        """
        Wide&Deep
        :@param feature_columns: 稀疏特征列的list
        :@param hidden_units: 隐层节点个数列表
        :@param activation: 激活函数
        :@param dnn_dropout: 随机失活概率
        :@param embed_reg: embedding正则化系数
        :@param w_r: w正则化系数
        """
        super(WideDeep, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.dnn_network = MyDNN(hidden_units, activation, dnn_dropout)
        self.linear = MyLinear(self.feature_length, w_r=w_r)
        self.final_dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](inputs[:, i])
                                  for i in range(inputs.shape[1])], axis=-1)
        x = sparse_embed  # (batch_size, field * embed_dim)
        
        # Wide部分
        wide_inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        wide_out = self.linear(wide_inputs)
        
        # Deep部分
        deep_out = self.dnn_network(x)
        deep_out = self.final_dense(deep_out)
        
        # 最终组合
        outputs = tf.nn.sigmoid(0.5 * wide_out + 0.5 * deep_out)
        return outputs

    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()