#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/23 17:10
# @Author  : 发电机转子
# @File    : gen_samples.py

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from config import parsers

opt = parsers()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)
torch.cuda.empty_cache()

# 导入模型
if opt.DATASET == 'SOLAR':
    HALF = 8.13/2  # 用于返归一化
    generator = torch.load('./model/generator_solar.pkl')
elif opt.DATASET == 'WIND':
    HALF = 16/2
    generator = torch.load('./model/generator_wind.pkl')
elif opt.DATASET == 'LOAD':
    HALF =7230.455/2
    generator = torch.load('./model/generator_load.pkl')
elif opt.DATASET == 'CARLOAD':
    HALF =4.45/2
    generator = torch.load('./model/generator_carload.pkl')

img_shape = (1, opt.img_size, opt.img_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

noise = Variable(Tensor(np.random.normal(0, 1, (8000, opt.latent_dim)))) #代表一列数据集的的长度，以一列为数据载入，而生成 1500负荷 9000光 105120风
out = generator(noise).view(-1, opt.img_size * opt.img_size)
out = out.cpu().detach().numpy()

sample = out * HALF + HALF  # 改成输入数据最大值的一半

Frame = pd.DataFrame(sample)


if opt.DATASET == 'SOLAR':
    Frame.to_csv('solarsamples.csv', header=None, index=None)
elif opt.DATASET == 'WIND':
    Frame.to_csv('windsamples.csv', header=None, index=None)
elif opt.DATASET == 'LOAD':
    Frame.to_csv('loadsamples.csv', header=None, index=None)
elif opt.DATASET == 'CARLOAD':
    Frame.to_csv('carloadsamples.csv', header=None, index=None)




fig, axs = plt.subplots(5, 5, dpi=300)
cnt = 0
for i in range(5):
    for j in range(5):
        axs[i, j].plot(sample[cnt, :288])
        axs[i, j].axis('off')
        cnt += 1
plt.show()
