from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras import *
import tensorflow as tf
import numpy as np
import args
import data


def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return tf.keras.layers.BatchNormalization
    elif norm == 'layer_norm':
        return tf.keras.layers.LayerNormalization


def conv_util(inputs, filters, kernel_size, strides=1, padding='same', activation='relu', norm='batch_norm'):
    Norm = _get_norm_layer(norm)
    output = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)(inputs)
    if norm == 'batch_norm':
        output = Norm(momentum=0.9)(output)
    else:
        output = Norm()(output)

    if activation == 'relu':
        output = tf.keras.layers.ReLU()(output)
    elif activation == 'none':
        return output
    else:
        output = tf.keras.layers.LeakyReLU(0.2)(output)
    return output


def res_block(inputs, filters, kernel_size, norm, activation='relu'):
    conv_1 = conv_util(inputs, filters // 2, kernel_size=1, norm=norm, activation=activation)
    conv_2 = conv_util(conv_1, filters // 2, kernel_size, norm=norm, activation=activation)
    conv_3 = conv_util(conv_2, filters, kernel_size=1, activation='none', norm=norm)
    output = tf.keras.layers.Add()([inputs, conv_3])
    if activation == 'relu':
        output = tf.keras.layers.ReLU()(output)
    elif activation == 'none':
        return output
    else:
        output = tf.keras.layers.LeakyReLU(0.2)(output)
    return output


def up_block(inputs, filters, kernel_size, strides=2, padding='same', norm='batch_norm'):
    Norm = _get_norm_layer(norm)
    output = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(inputs)
    if norm == 'batch_norm':
        output = Norm(momentum=0.9)(output)
    else:
        output = Norm()(output)
    output = tf.keras.layers.ReLU()(output)
    return output


def down_util(inputs, strides=2, padding='same'):
    output = tf.keras.layers.AveragePooling2D(2, strides, padding=padding)(inputs)
    # output = tf.keras.layers.LeakyReLU(0.2)(output)
    return output


# -------------------G-双线性插值上采样------------------------
def resize_up(inputs, filters, kernel_size, up_times=2,
              strides=1, padding='same', activation='relu', norm='batch_norm'):
    output = tf.keras.layers.UpSampling2D(up_times)(inputs)
    output = sep_conv_unit(output, filters, kernel_size, strides, padding, norm, activation)
    return output


def my_resblock_generator(input_shape, dim=32, kernel_size=3,
                          norm='batch_norm', name='my_resnet_generator'):
    inputs = tf.keras.Input(input_shape)
    output = conv_util(inputs, dim, kernel_size=1, norm='batch_norm')
    output = up_block(output, dim * 2, kernel_size=4, strides=1, padding='valid', norm='batch_norm')

    up_times = 3
    for up_time in range(up_times):
        dim = dim * 2
        output = res_block(output, dim, kernel_size, norm)
        # output = up_block(output, dim, kernel_size=kernel_size)
        output = tf.keras.layers.UpSampling2D(2)(output)

        if up_time == up_times - 1:
            output = conv_util(output, dim // 4, kernel_size=kernel_size)
        else:
            output = conv_util(output, dim * 2, kernel_size=kernel_size)

    output = conv_util(output, 32, kernel_size=1)
    output = conv_util(output, 3, kernel_size, norm='none', activation='none')
    output = tf.keras.layers.Activation('tanh')(output)
    return tf.keras.Model(inputs, output, name=name)


def my_resblock_discriminator(input_shape, dim=32, kernel_size=3, norm='none', name='my_resnet_discriminator'):
    inputs = tf.keras.Input(input_shape)
    output = conv_util(inputs, filters=dim * 2, kernel_size=5, strides=1, padding='same', norm=norm,
                       activation='leaky relu')

    down_times = 3
    for down_time in range(down_times):
        dim = dim * 2
        output = res_block(output, dim, kernel_size, norm=norm, activation='leaky relu')
        output = down_util(output)

        if down_time == down_times - 1:
            output = conv_util(output, filters=dim // 2, kernel_size=kernel_size, norm=norm, activation='leaky relu')
        else:
            output = conv_util(output, dim * 2, kernel_size=kernel_size, norm=norm, activation='leaky relu')

    q_output = Conv2D(filters=1, kernel_size=4, strides=1, padding='valid')(output)
    output_shape = output.shape
    return Model(inputs, outputs=[q_output, output], name=name), output_shape


def my_resblock_classifier(input_shape, n_classes, kernel_size=5, name='my_classifier_model'):
    inputs = tf.keras.Input(input_shape)
    dims = input_shape[-1]

    output = conv_util(inputs, filters=dims, kernel_size=kernel_size, norm='batch_norm', activation='leaky relu')
    # output = res_block(output, filters=dims * 2, kernel_size=kernel_size, norm='batch_norm', activation='leaky relu')
    output = down_util(output)

    # output = classifier_dense(output, dims)
    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    outputs = Dense(n_classes, activation='softmax')(output)
    return Model(inputs, outputs, name=name)


if __name__ == '__main__':
    model_, I = my_sep_discriminator((32, 32, 3), dim=64)
    model_.summary()
    print(I)
    model = my_resblock_generator(input_shape=(1, 1, 128 + 11), dim=64, kernel_size=5)
