#!/bin/bash

echo "开始训练"

Date=$(date +%Y%m%d%H%M)
export CUDA_VISIBLE_DEVICES=0

log_dir="logs/"

GPU=0

#python -m mask_test
#exit

nohup \
python -m train.train \
>> ./logs/console_$Date.log 2>&1 &
echo "启动完毕,在logs下查看日志！"
