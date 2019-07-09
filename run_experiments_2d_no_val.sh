#!/bin/sh

# 2d

for i in 1 2 3 4 5
do
python3 train_jonas_net_batch.py -name brats17_2d_pre_no_val_$i --batch_size 64 --patch_depth 1 --brats_train_year 17 --no_validation --seed $i
python3 train_jonas_net_batch.py -name brats17_2d_no_val_$i --batch_size 64 --patch_depth 1 --brats_train_year 17 --no_pretrained --no_validation --seed $i
done
