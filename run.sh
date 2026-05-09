#!/bin/bash

save_path=./ablation_checkpoints
data_path=./data/all_data.pkl
additive_json_path=./data/additives.json
searching_space_path=./data/searching_space_data_V2.csv

method=morgan
recommend_model=full_model
seed=42
usl_trials=10 # 10
epoch=200 # 200
pcl_trials=10 # 10
proto_epoch=300 # 300
EMA=True
use_decor_loss=True
use_topk=True
device=cuda:3


echo "======================================"
echo "Running recommendation pipeline"
echo "METHOD: $method"
echo "SEED: $seed"
echo "USE_EMA: $EMA"
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
    --device $device \
    --searching_space_path $searching_space_path \
    --additive_json_path $additive_json_path \
    --recommend_model $recommend_model
