#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 09:44
# @Author  : 发电机转子
# @File    : model.py

import torch
import numpy as np
from torch import nn

from config import parsers

opt = parsers()
img_shape = (opt.channels, opt.img_size, opt.img_size)


class Generator(nn.Module):
    """
    生成器模型
    """
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),                              #128维 映射到256维
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            #nn.Dropout(p=0.5),  #Dropout技术  torch.nn.Dropout待用
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        #img = self.keras.targeted.Dropout(rate=0.5)(img) #dropout技术 ，全链接网络的dropout技术   img=self.keras.layers.Dropout(rate=0.5)(img)
        img = img.view(img.shape[0], *img_shape)  # reshape成(1, 24, 24)
        return img


class Discriminator(nn.Module):
    """
    判别器模型
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),  # 全连接层 #torch.nn.Dropout(p=0.5), #Dropout技术

            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),  # 全连接层 #torch.nn.Dropout(p=0.5), #Dropout技术

            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  # 全连接层
            #nn.Dropout(p=0.5),  # Dropout技术
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        #img_flat = self.keras.targeted.Dropout(rate=0.5)(img_flat),  # Dropout技术 # Dropout技术
        validity = self.model(img_flat)
        return validity


if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()
    print(generator)





