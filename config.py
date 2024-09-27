#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 17:10
# @Author  : 发电机转子
# @File    : config.py

import argparse


def parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATASET", type=str, default='SOLAR', choices=['WIND', 'SOLAR', 'LOAD', 'CARLOAD'])
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches") #每次训练256个
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=24, help="size of each image dimension") #24*24=576， 输出一行576个数据，5分钟一个，表示生成两天(未来2880分钟的数据)。
                                                                                                      # 如果设置成100*100,生成的数据是10000个数据（未来50000分钟）

    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=5, help="interval betwen image samples")
    args = parser.parse_args(args=[])

    return args


if __name__ == "__main__":
    parsers()
