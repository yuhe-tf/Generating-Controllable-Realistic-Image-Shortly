from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from tensorflow.keras import *
from tensorflow import keras
import tensorflow as tf
import numpy as np
import imlib as im
import functools
import imageio
import glob
import time
import tqdm
import PIL
import os

import data
import args
import model
import module
from args import config
import loss

gen_image_per_row = 10

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Improve_ACGAN:
    def __init__(self):
        self.image_shape = config.image.shape
        self.num_classes = config.train.classes
        self.batch_size = config.train.batch_size
        self.epoch = config.train.epochs

        self.d_iter = config.train.d_iter
        self.data_name = config.train.data_name
        self.noise_dim = config.train.noise_dim
        self.loss_mode = config.train.loss_mode
        self.g_lr = config.train.generator_lr
        self.d_lr = config.train.discriminator_lr
        self.gradient_mode = config.train.gradient_mode
        self.g_p_weight = config.train.gradient_penalty_weight

        self.gen_one_img_save_path = config.train.gen_one_image_save_path
        self.gen_more_img_save_path = config.train.gen_more_image_save_path
        self.num_exam_gen = config.train.num_exam_gen

        # 0.用于保存训练过程中判别器、生成器的损失值
        self.record_npz_path = os.path.join('', '{}_record.npz'.format(self.data_name))
        # 1.设置虚假分类值
        self.fake_label = keras.utils.to_categorical([self.num_classes - 1] * self.batch_size, self.num_classes)

        # 2.搭建模型
        self.G = model.my_resblock_generator(input_shape=(1, 1, self.noise_dim + self.num_classes),
                                             kernel_size=3, dim=64)
        self.D, D_output_shape = model.my_resblock_discriminator(input_shape=self.image_shape, kernel_size=3,
                                                                 dim=64)
        self.C = model.my_resblock_classifier(input_shape=D_output_shape[1:], n_classes=self.num_classes,
                                              kernel_size=5)
        self.G.summary()
        self.D.summary()
        self.C.summary()

        # 3.设置损失函数与模型的优化器
        self.d_loss_fn, self.g_loss_fn = loss.get_adversarial_losses_fn(self.loss_mode)
        self.G_optimizer = keras.optimizers.Adam(learning_rate=self.d_lr, beta_1=0.5, beta_2=0.9)
        self.D_optimizer = keras.optimizers.Adam(learning_rate=self.g_lr, beta_1=0.5, beta_2=0.9)

        # 4. 保存与读取模型，保存模型
        self.checkpoint = tf.train.Checkpoint(G_optimizer=self.G_optimizer,
                                              D_optimizer=self.D_optimizer,
                                              G=self.G, D=self.D, C=self.C)
        self.manger = tf.train.CheckpointManager(self.checkpoint, directory='checkpoint', max_to_keep=1)
        self.load_model()

    def load_model(self):
        if self.manger.latest_checkpoint:
            self.checkpoint.restore(self.manger.latest_checkpoint)
            print(self.manger.latest_checkpoint)
        else:
            print('重新训练模型！')
        return

    # 6.定义模型辅助分类函数
    @staticmethod
    def classifier_loss(pred, label):
        cla_loss = keras.losses.categorical_crossentropy(y_pred=pred, y_true=label)
        return cla_loss

    # 6.1用于计算C模型分类的准确性
    @staticmethod
    def acc_fun(pred, label):
        return keras.metrics.categorical_accuracy(y_true=label, y_pred=pred)

    # 7.用于向量的拼接
    @staticmethod
    def concat_image_label(image, label):
        img_shapes = tf.shape(image)
        lab_shapes = tf.shape(label)
        lab = tf.reshape(label, [-1, 1, 1, lab_shapes[1]])
        y_shapes = tf.shape(lab)
        return tf.concat([image, lab * tf.ones([img_shapes[0], img_shapes[1], img_shapes[2], y_shapes[3]])], 3)

    # 8.生成器的训练步骤
    @tf.function
    def train_G(self, label):
        with tf.GradientTape() as t:
            # 用于生成模型的随机噪声
            z = tf.random.uniform((self.batch_size, 1, 1, self.noise_dim), maxval=1., minval=-1)
            # 将噪声于label拼接
            z = self.concat_image_label(z, label)

            x_fake = self.G(z, training=True)
            x_fake_d_logic, c_output = self.D(x_fake, training=True)
            c_output_ = self.C(c_output, training=True)

            G_loss = self.g_loss_fn(x_fake_d_logic) + self.classifier_loss(pred=c_output_, label=label)
            g_acc = self.acc_fun(c_output_, label)

        G_grad = t.gradient(G_loss, self.G.trainable_variables + self.C.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grad, self.G.trainable_variables + self.C.trainable_variables))
        return G_loss, g_acc

    # 9.判别器的训练步骤
    @tf.function
    def train_D(self, x_real, label):
        with tf.GradientTape() as t, tf.GradientTape() as c:
            z = tf.random.uniform((self.batch_size, 1, 1, self.noise_dim), minval=-1., maxval=1.)
            z = self.concat_image_label(z, label)

            x_fake = self.G(z, training=True)
            x_real_d_logic, real_c_output = self.D(x_real, training=True)
            x_fake_d_logic, fake_c_output = self.D(x_fake, training=True)
            x_real_d_loss, x_fake_d_loss = self.d_loss_fn(x_real_d_logic, x_fake_d_logic)

            gp = loss.gradient_penalty(functools.partial(self.D, training=True), x_real, x_fake,
                                       mode=self.gradient_mode)

            real_c_output_ = self.C(real_c_output, training=True)
            fake_c_output_ = self.C(fake_c_output, training=True)
            C_loss = self.classifier_loss(pred=real_c_output_, label=label) + self.classifier_loss(pred=fake_c_output_,
                                                                                                   label=self.fake_label)
            D_loss = (x_real_d_loss + x_fake_d_loss) + gp * self.g_p_weight + C_loss

            d_real_acc = self.acc_fun(pred=real_c_output_, label=label)
            d_fake_acc = self.acc_fun(pred=fake_c_output_, label=self.fake_label)

        D_grad = t.gradient(D_loss, self.D.trainable_variables + self.C.trainable_variables)
        self.D_optimizer.apply_gradients(zip(D_grad, self.D.trainable_variables + self.C.trainable_variables))

        return x_real_d_loss + x_fake_d_loss, d_real_acc, d_fake_acc

    def load_record(self):
        D_LOSS = []  # d的损失函数
        G_LOSS = []  # g损失函数
        G_ACC = []  # g生成图像被判断为真实图像的准确率
        D_ACC_F = []  # d判断生成图像为虚假图像的准确率
        D_ACC_R = []  # d判断真实图像为真实图像的分类准确率

        if os.path.exists(self.record_npz_path):
            with np.load(self.record_npz_path, allow_pickle=True) as file:
                D_LOSS = list(file.get('D_LOSS'))
                G_LOSS = list(file.get('G_LOSS'))
                G_ACC = list(file.get('G_ACC'))
                D_ACC_F = list(file.get('D_ACC_F'))
                D_ACC_R = list(file.get('D_ACC_R'))
        print('D_Loss shape:{}, G_Loss shape: {}'.format(len(D_LOSS), len(G_LOSS)))
        return D_LOSS, G_LOSS, G_ACC, D_ACC_F, D_ACC_R

    def train(self):
        # 3.获取训练数据
        train_data = data.get_data(batch_size=self.batch_size)
        module.check_path(self.gen_one_img_save_path)
        module.check_path(self.gen_more_img_save_path)

        tf.keras.backend.set_learning_phase(True)

        ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

        D_LOSS, G_LOSS, G_ACC, D_ACC_F, D_ACC_R = self.load_record()

        # d_iter discriminator训练d_iter次，generator训练一次
        test_label = [[i] * gen_image_per_row for i in range(self.num_classes - 1)]
        test_label = np.array([test_label[i][j] for i in range(self.num_classes - 1) for j in range(gen_image_per_row)])
        test_label = keras.utils.to_categorical(test_label, self.num_classes)
        z = tf.random.uniform((self.num_exam_gen, 1, 1, self.noise_dim), maxval=1., minval=-1)
        z = self.concat_image_label(z, test_label)

        # 1.使用预训练D 3000 次

        for epoch in tqdm.trange(self.epoch, desc='Epoch Loop'):
            if epoch < ep_cnt:
                continue

            # update epoch counter
            ep_cnt.assign_add(1)

            # train for an epoch
            for x_real, x_label in tqdm.tqdm(train_data, desc='Inner Epoch Loop', total=config.train.one_step):

                if len(x_label.shape) == 1:
                    x_label = keras.utils.to_categorical(x_label, CLASSES)

                d_loss, d_real_c, d_fake_c = self.train_D(x_real, x_label)

                D_LOSS.append(tf.reduce_mean(d_loss))
                D_ACC_F.append(tf.reduce_mean(d_fake_c))
                D_ACC_R.append(tf.reduce_mean(d_real_c))

                if self.D_optimizer.iterations.numpy() % self.d_iter == 0:
                    g_loss, g_acc = self.train_G(x_label)
                    G_LOSS.append(tf.reduce_mean(g_loss))
                    G_ACC.append(tf.reduce_mean(g_acc))

                # sample
                if self.G_optimizer.iterations.numpy() % 500 == 0:
                    x_fake = self.sample(z)
                    img = im.immerge(x_fake, n_rows=self.num_classes - 1).squeeze()
                    im.imwrite(img, os.path.join(self.gen_one_img_save_path,
                                                 'epoch_{:04d}_iter-{:09d}.jpg'.format(epoch,
                                                                                       self.G_optimizer.iterations.numpy())))
                    print('Epoch: {}, D_Loss: {:.2f}, D_ACC_F: {:.2f}, D_ACC_R: {:.2f}'.format(epoch + 1,
                                                                                               D_LOSS[-1],
                                                                                               D_ACC_F[-1],
                                                                                               D_ACC_R[-1]))
                    if self.G_optimizer.iterations.numpy() != 0:
                        print('G_LOSS: {:.2f}, G_ACC: {:.2f}'.format(G_LOSS[-1], G_ACC[-1]))

                    for i in range(config.train.test_gen):
                        test_input = tf.random.uniform((self.num_exam_gen, 1, 1, self.noise_dim), maxval=1., minval=-1.)
                        test_input = self.concat_image_label(test_input, test_label)
                        test_image = self.sample(test_input)
                        test_image = im.immerge(test_image, n_rows=self.num_classes - 1).squeeze()
                        im.imwrite(test_image, os.path.join(self.gen_more_img_save_path,
                                                            'iter-{:09d}-{:04d}.jpg'.format(
                                                                self.G_optimizer.iterations.numpy(), i)))

            # 保存
            np.savez(self.record_npz_path, D_LOSS=D_LOSS, G_LOSS=G_LOSS, G_ACC=G_ACC, D_ACC_F=D_ACC_F, D_ACC_R=D_ACC_R)
            self.manger.save(checkpoint_number=epoch)

    @tf.function
    def sample(self, z):
        # 10.生成测试图像样本
        return self.G(z, training=False)

    def gen_one_label(self, image_nums, label_index):
        # 12.指定生成一种标签的图像
        tf.keras.backend.set_learning_phase(True)

        n_rows = int(np.ceil(np.sqrt(image_nums)))
        test_label = [label_index] * image_nums
        test_label = keras.utils.to_categorical(test_label, self.num_classes)
        test_noise = tf.random.uniform((image_nums, 1, 1, self.noise_dim), minval=-1., maxval=1.)
        test_input = self.concat_image_label(test_noise, test_label)
        test_image = self.sample(test_input)
        test_image = im.immerge(test_image, n_rows=n_rows).squeeze()
        im.imwrite(test_image, os.path.join(self.gen_one_img_save_path,
                                            'label-{}.jpg'.format(label_index)))
        return

    # 13.定义测试函数
    def test(self, image_nums):
        # 1.生成所有不同种类的图像
        tf.keras.backend.set_learning_phase(True)
        test_label = [[i] * 10 for i in range(self.num_classes - 1)]
        test_label = np.array([test_label[i][j] for i in range(self.num_classes - 1) for j in range(10)])
        test_label = keras.utils.to_categorical(test_label, self.num_classes)
        for i in range(image_nums):
            test_input = tf.random.uniform((self.num_exam_gen, 1, 1, self.noise_dim), minval=-1., maxval=1.)
            test_input = self.concat_image_label(test_input, test_label)
            test_image = self.sample(test_input)
            test_image = im.immerge(test_image, n_rows=self.num_classes - 1).squeeze()
            im.imwrite(test_image, os.path.join(self.gen_more_img_save_path, 'iter-%d.jpg' % i))
        return


if __name__ == '__main__':
    TRAIN = True
    improve_acgan = Improve_ACGAN()
    if TRAIN:
        improve_acgan.train()
    else:
        improve_acgan.test(image_nums=10)
