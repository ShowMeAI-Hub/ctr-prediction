"""
DCN模型训练
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from model import DCN
from data_utils.process import create_criteo_dataset


if __name__ == '__main__':
    # 环境设定
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 超参数设定
    file = '../dataset/Criteo/train.txt'
    read_part = True
    sample_num = 200000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    # 构建数据集
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         embed_dim=embed_dim,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    
    # 模型构建
    model = DCN(feature_columns, hidden_units, dnn_dropout=dnn_dropout)
    model.summary()
    model.compile(loss=binary_crossentropy, 
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    
    # 模型训练
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # 测试集上验证效果
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])