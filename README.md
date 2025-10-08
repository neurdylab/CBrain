# CBrain
[MICCAI 2025] Code release for CBrain: Cross-Modal Learning for Brain Vigilance Detection in Resting-State fMRI

This codebase is developed based on [<a href="#ref1">1</a>][<a href="#ref2">2</a>].

## Checkpoint
Please download from: https://huggingface.co/alexandraChangLi/CBrain/tree/main.

## Environment Configs
The environment for data preprocessing and model training can be installed via the following commands:
```bash
conda create --name cbrain python=3.8.18
conda activate cbrain
conda install numpy
conda install pandas
conda install conda-forge::nibabel
conda install conda-forge::nilearn
conda install conda-forge::mne
conda install pytorch::pytorch
conda install conda-forge::tensorboard
conda install conda-forge::tensorboardx
pip install shap
pip install umap-learn
pip install plotly
pip install matplotlib
```
Package versions: \
python: 3.8.18 \
cuda: 12.4 \
numpy: 1.24.3 \
pandas: 2.0.3 \
nibabel: 5.2.1 \
nilearn: 0.10.4 \
mne: 1.6.1 \
pytorch: 2.4.1 \
tensorboard: 2.17.1 \
tensorboardx: 2.6.2.2 \
shap: 0.44.1 \
umap-learn: 0.5.7 \
plotly: 6.1.2 \
matplotlib: 3.7.2 

For a detailed version, please refer to: environment.yaml.

## Dataset Extraction

**Note:** EEGfMRI_VU dataset correspond to the training-testing dataset, and NIH dataset corresponds to the external validation dataset (ecr and ect). These datasets will be released.

To extract fMRI ROI time series from preprocessed data, run the following:
```bash
cd data_preprocessing/fmri_atlas
python datasetname_fmri_fit_atlas_batch_64.py
```
We use Matlab for converting the preprocessed EEG data into the .set format (data_preprocessing/eeg_filtering/datasetname_convertEEG_to_set_batch.m) and channel removal (data_preprocessing/eeg_filtering/datasetname_EEG_removechannels_batch.m). Required package: EEGLAB.
Please refer to NeuroBOLT[<a href="#ref3">3</a>] for data preprocessing details and [<a href="#ref4">4</a>] for vigilance ground truth extraction.

**Note from Authors:** The testing set is served as internal validation set where we pick the checkpoint that performs the best and report the metrics, and we use this selected checkpoint on the unseen external validation dataset. This approached is also applied to baselines compared.

## Model Training and Testing
Please modify main.py for the GPU index and the argument '--train_device' for CPU/GPU selection accordingly. 
```bash
bash scripts/train.sh
```
For redirecting outputs, please create the /logs folder and run:
```bash
nohup bash scripts/train.sh > logs/cbrain.txt &
```
Argument explanation:
```bash
#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
python main.py \
--model_name cbrain \ # model selected
--dataset_name eegfmri_vu \ # dataset selected
--train_device gpu \ # training device
--max_epoch 50 \
--warm_lr_epochs 20 \
--base_lr 7e-4 \
--loss_fmri_prediction_weight 0.5 \
--loss_fmri_label_weight 0.1 \
--loss_eeg_label_weight 0 \
--loss_fmri_eeg_weight 0.1 \
--checkpoint_dir outputs/cbrain \ # where the tensorboard files and the model checkpoints are stored
--dataset_num_workers 1 \
--log_metrics_every 5 \
--batchsize_per_gpu 32 \
--ngpus 1 \
# > logs/cbrain.txt \
```
Model checkpoints, final_eval.txt, and tensorboard files will be stored in --checkpoint_dir. To visualize training loss, run the following:
```bash
tensorboard --logdir=outputs/cbrain
```
For testing, run:
```bash
bash scripts/test.sh
``` 
Argument explanation:
```bash
#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
python main.py \
--model_name cbrain \
--dataset_name NIH_ecr \
--batchsize_per_gpu 32 \
--zero_shot \ # select one of the following: --zero_shot for zero_shot test on an unseen dataset (load the whole dataset for testing), --test_only for testing on testing set, --train_test for test the model's performance on the trained dataset in features visualization. 
--test_ckpt outputs/cbrain/checkpoint_best.pth \ # load the checkpoint that you would like to test
--test_logger outputs/cbrain/NIH_ecr \ # where tensorboard files save
> outputs/cbrain/NIH_ecr.txt \ # redirect output 
```

### Visualizations
To produce the visualization figures (Figure 2's plot of predictions, Figure 3's UMAP visualizations), first use the --test_visualize command to save the model predictions in:
```bash
bash scripts/test.sh
```
```bash
#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
python main.py \
--model_name cbrain \
--dataset_name NIH_ecr \
--test_visualize \ # include this argument to save the predictions for future visualization
--batchsize_per_gpu 32 \
--zero_shot \ # select one of the following: --zero_shot for zero_shot test on an unseen dataset (load the whole dataset for testing), --test_only for testing on testing set, --train_test for test the model's performance on the trained dataset in features visualization. 
--test_ckpt outputs/cbrain/checkpoint_best.pth \ # load the checkpoint that you would like to test
--test_logger outputs/cbrain/NIH_ecr \ # where tensorboard files save
> outputs/cbrain/NIH_ecr.txt \ # redirect output 
```
The predictions will be saved at outputs/cbrain/predictions_datasetname. Then, go to visualization_umaps.ipynb. Use the same environment for running the notebook. 

## Potential Questions
Please reach out to chang.li@vanderbilt.edu.

## References
<a id="ref1"></a>[[1] Misra I, Girdhar R, Joulin A. An end-to-end transformer model for 3d object detection[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 2906-2917.](https://openaccess.thecvf.com/content/ICCV2021/papers/Misra_An_End-to-End_Transformer_Model_for_3D_Object_Detection_ICCV_2021_paper.pdf)

<a id="ref2"></a>[[2] Lu Y, Xu C, Wei X, et al. Open-vocabulary point-cloud object detection without 3d annotation[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023: 1190-1199.](https://openaccess.thecvf.com/content/CVPR2023/papers/Lu_Open-Vocabulary_Point-Cloud_Object_Detection_Without_3D_Annotation_CVPR_2023_paper.pdf)

<a id="ref3"></a>[[3] Li Y, Lou A, Xu Z, et al. NeuroBOLT: Resting-state EEG-to-fMRI synthesis with multi-dimensional feature mapping[J]. Advances in neural information processing systems, 2024, 37: 23378-23405.](https://arxiv.org/abs/2410.05341)

<a id="ref4"></a>[[4] Pourmotabbed H, Martin C G, Goodale S E, et al. Multimodal state-dependent connectivity analysis of arousal and autonomic centers in the brainstem and basal forebrain[J]. Imaging Neuroscience, 2025, 3: IMAG. a. 91.](https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.91/131628)
