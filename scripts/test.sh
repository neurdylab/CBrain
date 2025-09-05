#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--model_name cbrain \
--dataset_name NIH_ecr \
--test_visualize \
--batchsize_per_gpu 32 \
--zero_shot \
--test_ckpt outputs/cbrain/checkpoint_best.pth \
--test_logger outputs/cbrain/NIH_ecr \
> outputs/cbrain/NIH_ecr.txt \