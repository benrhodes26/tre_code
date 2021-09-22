#!/bin/bash

for c in 1 3 5 6 7 8
do
    for i in 1 2
    do
        CUDA_VISIBLE_DEVICES=0 python build_bridges.py --config_path=gaussians/model/${c}_${i} > ${c}_run${i}.txt 2>&1
    done
done
