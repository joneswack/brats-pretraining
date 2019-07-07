#!/bin/sh

# 2d

for i in 1 2 3 4 5
do
python train_jonas_net_batch.py -name brats17_2d_pre_$i --batch_size 64 --patch_depth 1 --brats_train_year 17 --seed $i
python train_jonas_net_batch.py -name brats17_2d_$i --batch_size 64 --patch_depth 1 --brats_train_year 17 --no_pretrained --seed $i
done
