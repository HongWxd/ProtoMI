#!/bin/bash

training_methods=Dummy
use_D=False
use_SB=False
base_model=GCN
GPU=6

python SSL.py \
  --training_methods $training_methods \
  --use_D $use_D \
  --use_SB $use_SB \
  --base_model $base_model \
  --GPU $GPU
