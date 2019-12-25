import os


# 1. 保存模型在训练和测试时生成的图像
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return
