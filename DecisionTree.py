#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/1/14 16:06
# @Author  : 2012289 王麒翔
# @File    : DecisionTree.py

# 决策树
import warnings
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10

# 保存模型路径
MODEL_FILEPATH = 'saved_models/DT_cifar10.pvl'


# 获取CIFAR-10数据集的数据
def get_data():
    # 创建StandardScaler对象，用于数据的归一化和标准化
    scaler = StandardScaler()

    # x_train_original和y_train_original代表训练集的图像与标签, x_test_original与y_test_original代表测试集的图像与标签
    (x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()

    # 数据集分配：test代表测试集，train代表训练集
    x_test = x_test_original[5000:]
    y_test = y_test_original[5000:]
    x_train = x_train_original
    y_train = y_train_original

    # 这里把数据从uint类型转化为float32类型, 提高训练精度。
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 原始图像的像素灰度值为0-255，为了提高模型的训练精度，将数值归一化映射到0-1。
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 扁平化数据
    x_train = x_train.reshape((-1, 32 * 32 * 3))
    x_test = x_test.reshape((-1, 32 * 32 * 3))

    # 对数据进行标准化处理
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    # 将y转换为一维数组
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    return x_train, y_train, x_test, y_test


# 建立模型
def create_model():
    model = DecisionTreeClassifier()
    return model


# 训练模型
def train_model(model):
    # 获取所需数据集
    x_train, y_train, x_test, y_test = get_data()
    # 训练模型
    model.fit(x_train, y_train)
    # 模型保存
    joblib.dump(model, MODEL_FILEPATH)


# 在测试集上做预测并计算测试集准确率
def show_res(model):
    x_train, y_train, x_test, y_test = get_data()
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("决策树Accuracy:", accuracy)


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')
    print("下面进行决策树实验结果的展示...")
    # 根据模型是否创建采取不同方法
    try:
        model = joblib.load(MODEL_FILEPATH)
        print("决策树模型已创建，直接载入...")
    except:
        model = create_model()
        print("决策树模型未创建，首先创建...")
    train_model(model)
    show_res(model)
