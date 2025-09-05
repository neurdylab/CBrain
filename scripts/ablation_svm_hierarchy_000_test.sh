#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--model_name svm_hierarchy \
--dataset_name eegfmri_vu \
--test_only \
--test_visualize \
--batchsize_per_gpu 678 \
--test_ckpt outputs/ablation_svm_hierarchy_000/checkpoint_best.pth \
--test_logger outputs/ablation_svm_hierarchy_000/rbf_eegfmri_vu_test \
> outputs/ablation_svm_hierarchy_000/ablation_svm_hierarchy_000_rbf_eegfmri_vu_test.txt \