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


def res_block(inputs, filters, kernel_size, norm):
    conv_1 = conv_util(inputs, filters // 2, kernel_size=1, norm=norm)
    conv_2 = conv_util(conv_1, filters // 2, kernel_size, norm=norm)
    conv_3 = conv_util(conv_2, filters, kernel_size=1, activation='none', norm=norm)
    output = tf.keras.layers.Add()([inputs, conv_3])
    output = tf.keras.layers.ReLU()(output)
    return output


def up_block(inputs, filters, kernel_size, strides=2, padding='same'):
    Norm = _get_norm_layer('batch_norm')
    output = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(inputs)
    output = Norm()(output)
    output = tf.keras.layers.ReLU()(output)
    return output


def down_util(inputs, strides=2, padding='same'):
    output = tf.keras.layers.AveragePooling2D(2, strides, padding=padding)(inputs)
    output = tf.keras.layers.LeakyReLU(0.2)(output)
    return output


# ---------------------------Resblock Model---------------------------------------

def my_resblock_generator(input_shape, dim=64, kernel_size=3,
                          norm='batch_norm', name='my_resnet_generator'):
    inputs = tf.keras.Input(input_shape)
    output = conv_util(inputs, dim, kernel_size=1)
    output = up_block(output, dim * 2, kernel_size=4, strides=1, padding='valid')

    up_times = 3
    for up_time in range(up_times):
        dim = dim * 2
        output = res_block(output, dim, kernel_size, norm)
        output = tf.keras.layers.UpSampling2D(2)(output)

        if up_time == up_times - 1:
            output = conv_util(output, dim // 4, kernel_size=kernel_size)
        else:
            output = conv_util(output, dim * 2, kernel_size=kernel_size)

    output = conv_util(output, 64, kernel_size=1)
    output = conv_util(output, 3, kernel_size, norm='none', activation='none')
    output = tf.keras.layers.Activation('tanh')(output)
    return tf.keras.Model(inputs, output, name=name)


def my_resblock_discriminator(input_shape, dim=64, kernel_size=3, norm='none', name='my_resnet_discriminator'):
    inputs = tf.keras.Input(input_shape)
    output = conv_util(inputs, filters=dim * 2, kernel_size=5, strides=1, padding='same', norm=norm)

    down_times = 3
    for down_time in range(down_times):
        dim = dim * 2
        output = res_block(output, dim, kernel_size, norm)
        output = down_util(output)

        if down_time == down_times - 1:
            output = conv_util(output, filters=dim // 2, kernel_size=kernel_size, norm=norm)
        else:
            output = conv_util(output, dim * 2, kernel_size=kernel_size, norm=norm)

    q_output = Conv2D(filters=1, kernel_size=4, strides=1, padding='valid')(output)
    output_shape = output.shape
    return Model(inputs, outputs=[q_output, output], name=name), output_shape


def my_resblock_classifier(input_shape, n_classes, kernel_size=5, name='my_classifier_model'):
    inputs = tf.keras.Input(input_shape)
    dims = input_shape[-1]

    output = conv_util(inputs, filters=dims, kernel_size=kernel_size, norm='batch_norm', activation='leaky relu')
    output = down_util(output)

    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    outputs = Dense(n_classes, activation='softmax')(output)
    return Model(inputs, outputs, name=name)


'''


def my_resblock_generator(input_shape, dim=32, kernel_size=3,
                          norm='batch_norm', name='my_resnet_generator'):
    inputs = tf.keras.Input(input_shape)
    output = conv_util(inputs, dim, kernel_size=1)
    output = resize_up(output, dim * 2, kernel_size=3, up_times=4)

    up_times = 3
    for up_time in range(up_times):
        dim = dim * 2
        output = res_block(output, dim, kernel_size, norm)
        if up_time == up_times - 1:
            output = resize_up(output, dim // 4, kernel_size=kernel_size)
        else:
            output = resize_up(output, dim * 2, kernel_size=kernel_size)

    output = conv_util(output, dim, kernel_size=1)
    output = conv_util(output, 3, kernel_size, norm='none', activation='none')
    output = tf.keras.layers.Activation('tanh')(output)
    return tf.keras.Model(inputs, output, name=name)


def my_resblock_discriminator(input_shape, dim=32, kernel_size=3, norm='none', name='my_resnet_discriminator'):
    inputs = tf.keras.Input(input_shape)
    output = conv_util(inputs, filters=dim * 2, kernel_size=5, strides=1, padding='same', norm=norm)

    down_times = 3
    for down_time in range(down_times):
        dim = dim * 2
        output = res_block(output, dim, kernel_size, norm)
        output = down_util(output)
        if down_time == down_times - 1:
            output = conv_util(output, filters=dim // 2, kernel_size=kernel_size, norm=norm)
        else:
            output = conv_util(output, dim * 2, kernel_size=kernel_size, norm=norm)

    q_output = Conv2D(filters=1, kernel_size=4, strides=1, padding='valid')(output)
    output_shape = output.shape
    return Model(inputs, outputs=[q_output, output], name=name), output_shape


def my_resblock_classifier(input_shape, n_classes, kernel_size=5, name='my_classifier_model'):
    inputs = tf.keras.Input(input_shape)
    dims = input_shape[-1]

    output = res_block(inputs, filters=dims, kernel_size=kernel_size, norm='batch_norm')
    output = down_util(output)

    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    outputs = Dense(n_classes, activation='softmax')(output)
    return Model(inputs, outputs, name=name)
'''

if __name__ == '__main__':
    model_, _ = my_resblock_discriminator((32, 32, 3))
    model_.summary()
    g = my_resblock_generator((1, 1, 128))
    g.summary()
