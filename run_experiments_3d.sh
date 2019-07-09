#!/bin/sh

# 3d

for i in 1 2 3 4 5
do
python3 train_jonas_net_batch.py -name brats17_3d_pre_$i --batch_size 24 --patch_depth 24 --brats_train_year 17 --seed $i
python3 train_jonas_net_batch.py -name brats17_3d_$i --batch_size 24 --patch_depth 24 --brats_train_year 17 --no_pretrained --seed $i
done
