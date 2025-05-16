
#!/bin/bash

python get_train_data.py

python train_cwq_ranker.py \
    --POS_NUM 10 \
    --NEG_NUM 15 \
    --num_train_epochs 2 \
    --out_dir "cwq_ranker" \
    --hard_pos 1 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 32 \
    --margin 0.8\
    > cwq_ranker.log 2>&1



