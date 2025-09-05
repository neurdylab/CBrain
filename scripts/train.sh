#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--model_name cbrain \
--dataset_name eegfmri_vu \
--train_device gpu \
--max_epoch 50 \
--warm_lr_epochs 20 \
--base_lr 7e-4 \
--loss_fmri_prediction_weight 0.5 \
--loss_fmri_label_weight 0.1 \
--loss_eeg_label_weight 0 \
--loss_fmri_eeg_weight 0.1 \
--checkpoint_dir outputs/cbrain \
--dataset_num_workers 1 \
--log_metrics_every 5 \
--batchsize_per_gpu 32 \
--ngpus 1 \
# > logs/cbrain.txt \