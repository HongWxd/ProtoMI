#!/bin/bash

save_path=./ablation_checkpoints
data_path=./data/all_data.pkl
method=morgan
seed=42
usl_trials=10 # 10
epoch=200 # 200
pcl_trials=10 # 10
proto_epoch=300 # 300
EMA=True
use_decor_loss=True
use_topk=False
device=cuda:3


echo "======================================"
echo "Running recommendation pipeline"
echo "METHOD: $method"
echo "SEED: $seed"
echo "EMA: $EMA"
echo "USE_DECOR_LOSS: $use_decor_loss"
echo "USE_TOPK: $use_topk"
echo "DEVICE: $device"
echo "======================================"

python run_main.py \
    --method $method \
    --random_state $seed \
    --data_path $data_path \
    --save_path $save_path \
    --usl_trials $usl_trials \
    --pcl_trials $pcl_trials \
    --epoch $epoch \
    --proto_epoch $proto_epoch \
    --EMA $EMA \
    --use_decor_loss $use_decor_loss \
    --use_topk $use_topk \
    --device $device
