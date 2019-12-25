from easydict import EasyDict as edict
import pprint
import os

config = edict()

# 1.图像的信息配置
config.image = edict()
config.image.height = 32
config.image.width = 32
config.image.channel = 3
config.image.shape = [config.image.height, config.image.width, config.image.channel]

# 2.训练配置
config.train = edict()
config.train.d_iter = 5  # 判别器训练5次，生成器训练1次
config.train.d_norm = None
config.train.loss_mode = 'wgan'
config.train.gradient_mode = 'wgan-gp'
config.train.test_gen = 50
config.train.classifier_lr = 0.001
config.train.generator_lr = 0.0002
config.train.discriminator_lr = 0.0006
config.train.gradient_penalty_weight = 10

config.train.epochs = 200
config.train.classes = 10 + 1  # 五个真实类别一个虚假类别
config.train.batch_size = 100
config.train.noise_dim = 128
config.train.iteration = 50000
config.train.image_count = 60000  # 有多少张训练图像
config.train.data_name = 'cifar10'
config.train.num_exam_gen = (config.train.classes - 1) * 10
config.train.one_step = config.train.image_count // config.train.batch_size
config.train.gen_one_image_save_path = os.path.join('gen_samples', 'one_samples')
config.train.gen_more_image_save_path = os.path.join('gen_samples', 'more_samples')
config.train.gan_shape = [config.image.height, config.image.width, config.image.channel + config.train.classes]

# 3.打印配置信息
pprint.pprint(config)
