#!/bin/bash

python get_train_data.py

python train_ranker.py \
    --POS_NUM 10 \
    --NEG_NUM 8 \
    --num_train_epochs 3 \
    --out_dir "webqsp_ranker" \
    --hard_pos 1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 32 \
    --margin 1.0\
    > webqsp_ranker.log 2>&1


