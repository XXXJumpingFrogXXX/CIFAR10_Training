#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/5 22:45
# @Author  : 2012289 王麒翔
# @File    : FullyConnectedNN.py

# 全连接网络
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.datasets import cifar10
from keras.utils import np_utils

# 保存模型路径
MODEL_FILEPATH = 'saved_models/FCNN_cifar10.h5'
# 训练epoch
TRAIN_EPOCH = 200
# 批大小
BATCH_SIZE = 256


# 获取CIFAR-10数据集的数据
def get_data():
    # x_train_original和y_train_original代表训练集的图像与标签, x_test_original与y_test_original代表测试集的图像与标签
    (x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()

    # 数据集分配：val代表验证集 test代表测试集 train代表训练集
    x_val = x_test_original[:5000]
    y_val = y_test_original[:5000]
    x_test = x_test_original[5000:]
    y_test = y_test_original[5000:]
    x_train = x_train_original
    y_train = y_train_original

    # 这里把数据从uint类型转化为float32类型, 提高训练精度。
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    # 原始图像的像素灰度值为0-255，为了提高模型的训练精度，将数值归一化映射到0-1。
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0

    # 图像标签一共有10个类别即0-9，这里将其转化为独热编码（One-hot）向量
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    y_test = np_utils.to_categorical(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


# 建立神经网络模型
def create_model():
    model = Sequential()

    # 网络结构 模型图可见于报告对应部分
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add((tf.keras.layers.Dropout(0.25)))

    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add((tf.keras.layers.Dropout(0.25)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # 打印网络结构
    print(model.summary())
    return model


# 训练模型
def train_model(model):
    # 获取所需数据集
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    # 编译网络（定义损失函数、优化器、评估指标）
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 开始网络训练（定义训练数据与验证数据、定义训练代数，定义训练批大小）
    train_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=TRAIN_EPOCH,
                              batch_size=BATCH_SIZE, verbose=1)
    # 模型保存
    model.save(MODEL_FILEPATH)
    # 返回训练历史
    return train_history


# 可视化展示训练历史
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


# 在测试集上做预测并计算测试集准确率
def show_res(model):
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    score = model.evaluate(x_test, y_test)
    print('全连接神经网络Loss:', score[0])
    print('全连接神经网络Accuracy:', score[1])


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')
    # 根据模型是否创建采取不同方法
    try:
        model = tf.keras.models.load_model(MODEL_FILEPATH)
        print("全连接网络模型已创建，直接载入...")
    except:
        model = create_model()
        print("全连接网络模型未创建，首先创建...")
    # 训练模型并获取训练历史
    train_history = train_model(model)
    # 展示训练历史
    show_train_history(train_history, 'accuracy', 'val_accuracy')
    show_train_history(train_history, 'loss', 'val_loss')
    # 展示模型结果
    show_res(model)
