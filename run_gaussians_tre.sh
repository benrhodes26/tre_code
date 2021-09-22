#!/bin/bash

for c in 0 2 4
do
    for i in 1 2
    do
        CUDA_VISIBLE_DEVICES=1 python build_bridges.py --config_path=gaussians/model/${c}_${i} > ${c}_run${i}.txt
    done
done