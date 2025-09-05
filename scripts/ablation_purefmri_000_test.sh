#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--model_name cbrain_purefmri \
--dataset_name NIH_ect \
--batchsize_per_gpu 32 \
--zero_shot \
--test_ckpt outputs/ablation_purefmri_000/checkpoint_best.pth \
--test_logger outputs/ablation_purefmri_000/NIH_ect_32 \
> outputs/ablation_purefmri_000/NIH_ect_zeroshot_32.txt \