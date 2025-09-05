#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--model_name random_guess \
--dataset_name NIH_ecr \
--zero_shot \
--batchsize_per_gpu 32 \
--test_ckpt outputs/ablation_random_guess_000/checkpoint_best.pth \
--test_logger outputs/ablation_random_guess_000/NIH_ecr_zeroshot \
> outputs/ablation_random_guess_000/NIH_ecr_zeroshot.txt \