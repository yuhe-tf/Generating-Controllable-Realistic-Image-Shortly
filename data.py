from tensorflow.keras import datasets, utils, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras import *
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2 as cv
import pathlib
import time
import tqdm
import os

import args
from args import config


# 1.获取训练数据集
def get_data(batch_size, repeat=1):
    if config.train.data_name == 'cifar10':
        return cifar_data(batch_size, repeat)


# 2.样本归一化
def data_pro(image, label):
    image = tf.cast(image, np.float32)
    image = (image / 255.0 - 0.5) * 2  # 使所有的像素值归一到[-1, 1]之间
    return image, label


# 3.cifar10样本获取
def cifar_data(batch_size, repeat):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = np.concatenate((x_train, x_test), 0)
    y_train = np.concatenate((y_train, y_test), 0)

    y_train = keras.utils.to_categorical(y_train, config.train.classes)

    cifar = tf.data.Dataset.from_tensor_slices((x_train, y_train)). \
        shuffle(5000).batch(batch_size, drop_remainder=True).repeat(repeat)
    cifar = cifar.map(data_pro)

    return cifar


if __name__ == '__main__':
    pass
