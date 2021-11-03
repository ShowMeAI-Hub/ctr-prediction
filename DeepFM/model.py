"""
DeepFM模型实现
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input, Layer


class MyFM(Layer):
    """
    Wide部分
    """
    def __init__(self, feature_length, w_r=1e-6):
        """
        
        :@param feature_length: 特征长度
        :@param w_r: w正则化系数
        """
        super(FM, self).__init__()
        self.feature_length = feature_length
        self.w_r = w_r

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_r),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        """
        :@param inputs: 1个字典，维度为 `(batch_size, {'sparse_inputs', 'embed_inputs'})`:
          其中sparse_inputs是一个维度为 `(batch_size, sum(field_num))`的2D张量
             embed_inputs是一个维度为 `(batch_size, fields, embed_dim)`的3D张量
        """
        sparse_inputs, embed_inputs = inputs['sparse_inputs'], inputs['embed_inputs']
        
        # 一阶项
        first_order = tf.reduce_sum(tf.nn.embedding_lookup(self.w, sparse_inputs), axis=1)  # (batch_size, 1)
        
        # 二阶项
        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        return first_order + second_order


class MyDNN(Layer):
    """
    Deep部分
    """
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0.):
        """
        DNN part
        :@param hidden_units: 形如`[unit1, unit2,...,]`的隐层神经元数量列表
        :@param activation: 激活函数
        :@param dnn_dropout: 随机失活概率
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class DeepFM(Model):
	def __init__(self, feature_columns, hidden_units=(200, 200, 200), dnn_dropout=0.,
				 activation='relu', fm_w_r=1e-6, embed_reg=1e-6):
		"""
		DeepFM
		:@param feature_columns: 稀疏特征列表
		:@param hidden_units: 隐层神经元个数列表
		:@param dnn_dropout: 随机失活概率
		:@param activation: 激活函数
		:@param fm_w_r: FM中w的正则化系数
		:@param embed_reg: embedding正则化系数
		"""
		super(DeepFM, self).__init__()
		self.sparse_feature_columns = feature_columns
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
		self.embed_dim = self.sparse_feature_columns[0]['embed_dim']  # all sparse features have the same embed_dim
		self.fm = MyFM(self.feature_length, fm_w_r)
		self.dnn = MyDNN(hidden_units, activation, dnn_dropout)
		self.dense = Dense(1, activation=None)

	def call(self, inputs, **kwargs):
		sparse_inputs = inputs
		# embedding
		sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)  # (batch_size, embed_dim * fields)
		# wide
		sparse_inputs = sparse_inputs + tf.convert_to_tensor(self.index_mapping)
		wide_inputs = {'sparse_inputs': sparse_inputs,
					   'embed_inputs': tf.reshape(sparse_embed, shape=(-1, sparse_inputs.shape[1], self.embed_dim))}
		wide_outputs = self.fm(wide_inputs)  # (batch_size, 1)
		# deep
		deep_outputs = self.dnn(sparse_embed)
		deep_outputs = self.dense(deep_outputs)  # (batch_size, 1)
		
		# 汇总
		outputs = tf.nn.sigmoid(tf.add(wide_outputs, deep_outputs))
		return outputs

	def summary(self):
		sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
		Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
