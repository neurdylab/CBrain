#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--model_name random_guess \
--dataset_name eegfmri_vu \
--train_device gpu \
--max_epoch 1 \
--warm_lr_epochs 20 \
--base_lr 7e-4 \
--loss_fmri_prediction_weight 0.1 \
--loss_fmri_label_weight 0.1 \
--loss_eeg_label_weight 0 \
--loss_fmri_eeg_weight 0.1 \
--checkpoint_dir outputs/ablation_random_guess_000 \
--dataset_num_workers 1 \
--log_metrics_every 5 \
--batchsize_per_gpu 32 \
--ngpus 1 \
# > logs/ablation_random_guess_000_train.txt \
