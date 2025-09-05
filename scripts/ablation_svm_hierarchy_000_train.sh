#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--model_name svm_hierarchy \
--dataset_name eegfmri_vu \
--train_device cpu \
--max_epoch 1 \
--base_lr 7e-4 \
--loss_fmri_prediction_weight 0 \
--loss_fmri_label_weight 0 \
--loss_eeg_label_weight 0 \
--loss_fmri_eeg_weight 0.1 \
--checkpoint_dir outputs/ablation_svm_hierarchy_000 \
--dataset_num_workers 1 \
--log_metrics_every 1 \
--batchsize_per_gpu 2599 \ 
--ngpus 1 \
# > logs/ablation_svm_hierarchy_000_train.txt \
# Note: load all of the samples in one training batch