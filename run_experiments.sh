#!/bin/sh

# 2d
python train_jonas_net_batch.py -name brats17_2d_pre --batch_size 64 --patch_depth 1 --brats_train_year 17 --no_gpu --seed 0
python train_jonas_net_batch.py -name brats17_2d --batch_size 64 --patch_depth 1 --brats_train_year 17 --no_gpu --no_pretrained --seed 0

python train_jonas_net_batch.py -name brats17_2d_pre --batch_size 64 --patch_depth 1 --brats_train_year 17 --no_gpu --no_validation --seed 0
python train_jonas_net_batch.py -name brats17_2d --batch_size 64 --patch_depth 1 --brats_train_year 17 --no_gpu --no_pretrained --no_validation --seed 0

#3d

