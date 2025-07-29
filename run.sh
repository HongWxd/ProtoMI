#!/bin/bash

training_methods=Dummy
use_D=True
use_SB=True
base_model=GINE
GPU=5

python SSL.py \
  --training_methods $training_methods \
  --use_D $use_D \
  --use_SB $use_SB \
  --base_model $base_model \
  --GPU $GPU
