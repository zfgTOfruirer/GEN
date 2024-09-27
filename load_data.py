#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : 发电机转子
# @File    : load_data.py

import pandas as pd
import numpy as np
import os
import torch
from torch.utils import data
from config import parsers

opt = parsers()


def wind_dataloader(root):
    trainset = 'wind.csv'

    rows = pd.read_csv(os.path.join(root, trainset), header=None)
    rows = np.array(rows, dtype=np.float32)
    # rows = rows[:, :32]
    # max_rows = np.max(rows)
    # rows = rows / max_rows
    half_rows = np.max(rows) / 2
    rows = (rows - half_rows) / half_rows

    trX = None
    for x in range(rows.shape[1]):
        train = rows[:, x].reshape(-1, opt.img_size*opt.img_size)
        if trX is None:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)

    trX = trX.reshape(-1, 1, opt.img_size, opt.img_size)
    trX = torch.tensor(trX)
    train_ids = data.TensorDataset(trX)
    dataloader = data.DataLoader(train_ids, batch_size=opt.batch_size, shuffle=True)
    return dataloader

def solar_dataloader(root):
    trainset = 'solar.csv'

    rows = pd.read_csv(os.path.join(root, trainset), header=None)
    rows = np.array(rows, dtype=np.float32)

    half_rows = np.max(rows) / 2
    rows = (rows - half_rows) / half_rows

    trX = []
    for x in range(rows.shape[1]):
        train = rows[:104832, x].reshape(-1, opt.img_size * opt.img_size)#104832表示24*24的整数倍
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)

    trX = trX.reshape(-1, 1, opt.img_size, opt.img_size)
    trX = torch.tensor(trX)
    train_ids = data.TensorDataset(trX)
    dataloader = data.DataLoader(train_ids, batch_size=256, shuffle=True)
    return dataloader

def load_dataloader(root):
    trainset = 'load.csv'

    rows = pd.read_csv(os.path.join(root, trainset), header=None)
    rows = np.array(rows, dtype=np.float32)

    half_rows = np.max(rows) / 2
    rows = (rows - half_rows) / half_rows

    trX = []
    for x in range(rows.shape[1]):
        train = rows[:11520, x].reshape(-1, opt.img_size * opt.img_size)
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)

    trX = trX.reshape(-1, 1, opt.img_size, opt.img_size)
    trX = torch.tensor(trX)
    train_ids = data.TensorDataset(trX)
    dataloader = data.DataLoader(train_ids, batch_size=256, shuffle=True)
    return dataloader

def carload_dataloader(root):
    trainset = 'carload.csv'

    rows = pd.read_csv(os.path.join(root, trainset), header=None)
    rows = np.array(rows, dtype=np.float32)
    # rows = rows[:, :32]
    # max_rows = np.max(rows)
    # rows = rows / max_rows
    half_rows = np.max(rows) / 2
    rows = (rows - half_rows) / half_rows

    trX = None
    for x in range(rows.shape[1]):
        train = rows[:1152, x].reshape(-1, opt.img_size*opt.img_size)
        if trX is None:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)

    trX = trX.reshape(-1, 1, opt.img_size, opt.img_size)
    trX = torch.tensor(trX)
    train_ids = data.TensorDataset(trX)
    dataloader = data.DataLoader(train_ids, batch_size=opt.batch_size, shuffle=True)
    return dataloader





if __name__ == "__main__":
    root = './datasets'
    train_ids = solar_dataloader(root)
    x = next(iter(train_ids))
    for i, imgs in enumerate(train_ids):
        print(i)
        print(imgs[0].shape)
