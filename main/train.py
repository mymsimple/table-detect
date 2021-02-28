#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020

@author: chineseocr
"""
import sys
sys.path.append('.')
from main.table_line import model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from glob import glob
from utils.image import gen
import utils.config as conf
import time
import logging

logger = logging.getLogger("Table-Detect")

CUDA_VISIBLE_DEVICES=0


def train():
    #logger.info("所有的参数：%r", args)

    timestamp_s = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    tb_log_name = conf.DIR_TBOARD + "/" + timestamp_s

    filepath = 'models/table-line-fine.h5'  ##模型权重存放位置

    paths = glob('data/train/*.json')  ##table line dataset label with labelme
    print("加载数据:", len(paths))

    Checkpointer = ModelCheckpoint(filepath=filepath,
                                   monitor='loss',
                                   verbose=0,
                                   save_weights_only=True,
                                   save_best_only=True)
    EarlyStop = EarlyStopping(monitor='loss',
                              patience=2, verbose=1, mode='auto')
    # 减小学习率
    Reduce = ReduceLROnPlateau(monitor='loss',
                               factor=0.1,
                               patience=5,
                               verbose=1,
                               mode='auto',
                               epsilon=0.0001,
                               cooldown=0,
                               min_lr=0)

    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])

    trainP, testP = train_test_split(paths, test_size=0.12)
    logger.info('total:%r, train:%r, test:%r', len(paths), len(trainP), len(testP))

    batchsize = 4
    trainloader = gen(trainP, batchsize=batchsize, linetype=2)
    testloader = gen(testP, batchsize=batchsize, linetype=2)

    model.fit_generator(generator=trainloader,
                        steps_per_epoch=max(1, len(trainP) // batchsize),
                        callbacks=[TensorBoard(log_dir=tb_log_name), Checkpointer, EarlyStop, Reduce],
                        use_multiprocessing=True,
                        epochs=150,
                        workers=10,
                        validation_data=testloader,
                        validation_steps=max(1, len(testP) // batchsize)
                        )


if __name__=='__main__':
    #log.init()
    #args = config.init_args()
    train()
