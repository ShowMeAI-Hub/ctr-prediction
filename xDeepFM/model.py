"""
xDeepFM模型TensorFlow2.X构建
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Flatten, Dense, Input, Dropout, Layer


class MyDNN(Layer):
    def __init__(self, hidden_units, dnn_dropout=0., dnn_activation='relu'):
        """
        DNN部分
        :@param hidden_units: 隐层节点个数列表
        :@param dnn_dropout: 随机失活的概率
        :@param dnn_activation: 激活函数
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=dnn_activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


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


class CIN(Layer):
    def __init__(self, cin_size, l2_r=1e-4):
        """
        CIN结构实现
        :@param cin_size: 格式为[H_1, H_2 ,..., H_k]的列表
        :@param l2_r: L2正则化系数
        """
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_r = l2_r

    def build(self, input_shape):
        # embedding fields的数量
        self.embedding_nums = input_shape[1]
        # CIN数量列表
        self.field_nums = [self.embedding_nums] + self.cin_size
        # filters
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_normal',
                regularizer=l2(self.l2_r),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs, **kwargs):
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        # split dimension 2 for calculation convenience
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)  # dim * (None, field_nums[0], 1)
        for idx, size in enumerate(self.cin_size):
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)  # dim * (None, filed_nums[i], 1)

            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)  # (dim, None, field_nums[0], field_nums[i])

            result_2 = tf.reshape(result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])

            result_3 = tf.transpose(result_2, perm=[1, 0, 2])  # (None, dim, field_nums[0] * field_nums[i])

            result_4 = tf.nn.conv1d(input=result_3, 
                                    filters=self.cin_W['CIN_W_' + str(idx)], 
                                    stride=1,
                                    padding='VALID')

            result_5 = tf.transpose(result_4, perm=[0, 2, 1])  # (None, field_num[i+1], dim)

            hidden_layers_results.append(result_5)

        final_results = hidden_layers_results[1:]
        result = tf.concat(final_results, axis=1)  # (None, H_1 + ... + H_K, dim)
        result = tf.reduce_sum(result,  axis=-1)  # (None, dim)

        return result


class xDeepFM(Model):
    def __init__(self, feature_columns, hidden_units, cin_size, dnn_dropout=0, dnn_activation='relu',
                 embed_reg=1e-6, cin_reg=1e-6, w_r=1e-6):
        """
        xDeepFM
        :@param feature_columns: 稀疏特征列表
        :@param hidden_units: 隐层神经元个数列表
        :@param cin_size: CIN层数量列表
        :@param dnn_dropout: 随机失活概率
        :@param dnn_activation: 激活函数
        :@param embed_reg: embedding正则化系数
        :@param cin_reg: cin正则化系数
        :@param w_r: w的正则化系数
        """
        super(xDeepFM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.linear = Linear(self.feature_length, w_r)
        self.cin = CIN(cin_size=cin_size, l2_r=cin_reg)
        self.dnn = DNN(hidden_units=hidden_units, dnn_dropout=dnn_dropout, dnn_activation=dnn_activation)
        self.cin_dense = Dense(1)
        self.dnn_dense = Dense(1)
        self.bias = self.add_weight(name='bias', shape=(1, ), initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        # Linear
        linear_inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        linear_out = self.linear(linear_inputs)  # (batch_size, 1)
        
        # cin
        embed = [self.embed_layers['embed_{}'.format(i)](inputs[:, i]) for i in range(inputs.shape[1])]
        embed_matrix = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        cin_out = self.cin(embed_matrix)  # (batch_size, dim)
        cin_out = self.cin_dense(cin_out)  # (batch_size, 1)
        
        # dnn
        embed_vec = tf.reshape(embed_matrix, shape=(-1, embed_matrix.shape[1] * embed_matrix.shape[2]))
        dnn_out = self.dnn(embed_vec)
        dnn_out = self.dnn_dense(dnn_out)  # (batch_size, 1))
        
        # output
        output = tf.nn.sigmoid(linear_out + cin_out + dnn_out + self.bias)
        return output

    def summary(self):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()