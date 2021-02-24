#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020

@author: chineseocr
"""
import sys
sys.path.append('.')
from table_line import model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from glob import glob
from image import gen
import logging

logger = logging.getLogger("Train")

CUDA_VISIBLE_DEVICES=0

def train():
    #logger.info("所有的参数：%r", args)

    filepath = 'models/table-line-fine.h5'  ##模型权重存放位置
    paths = glob('data/train/*.json')  ##table line dataset label with labelme
    logger.info("加载数据：%r条",len(paths))
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='loss', verbose=0, save_weights_only=True,
                                   save_best_only=True)
    rlu = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=0, mode='auto', cooldown=0, min_lr=0)
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])

    trainP, testP = train_test_split(paths, test_size=0.1)
    logger.info('total:%r, train:%r, test:%r', len(paths),len(trainP),len(testP))

    batchsize = 4
    trainloader = gen(trainP, batchsize=batchsize, linetype=1)
    testloader = gen(testP, batchsize=batchsize, linetype=1)

    model.fit_generator(trainloader,
                        steps_per_epoch=max(1, len(trainP) // batchsize),
                        callbacks=[checkpointer],
                        validation_data=testloader,
                        validation_steps=max(1, len(testP) // batchsize),
                        epochs=30)
                        #workers=6)


if __name__=='__main__':
    #log.init()
    #args = config.init_args()
    train()
