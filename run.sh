#!/bin/bash

training_methods=SSL
use_D=True
use_SB=True
base_model=GCN
GPU=6

python SSL.py \
  --training_methods $training_methods \
  --use_D $use_D \
  --use_SB $use_SB \
  --base_model $base_model \
  --GPU $GPU
