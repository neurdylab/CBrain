# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import datetime
import logging
import math
import time
import sys
import os
import numpy as np
from utils.ac_re_calculator import ACRECalculator
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
)


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
):
    acre_calculator = ACRECalculator(
        dataset_config=dataset_config,
    )

    curr_iter = curr_epoch * len(dataset_loader)

    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)
        
        optimizer.zero_grad()
        inputs = {
            "fmri": batch_data_label["fmri"].float(),
            "eeg": batch_data_label["eeg"].float(),
            "bad_tr": batch_data_label["bad_tr"].float(),
            "eeg_index": batch_data_label["eeg_index"].float(),
        }
        outputs = model(inputs)

        loss, loss_dict = criterion(outputs, batch_data_label)
        loss_reduced = all_reduce_average(loss)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss = loss.float()
        
        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)
        
        loss.requires_grad_(True)

        if args.l2_loss:
            lambda_l2 = 1e-4 
            l2_loss = lambda_l2 * sum(p.pow(2.0).sum() for p in model.parameters())
            l2_loss.requires_grad_(True)
            total_loss = loss + l2_loss
            total_loss.backward()
        else:
            loss.backward()

        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            acre_calculator.step_meter(outputs, batch_data_label)

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")

        curr_iter += 1
        barrier()
        
    return acre_calculator

@torch.no_grad()
def visualize(dataset_loader, dataset_config, model, logger, args):
    fmri_feats_total = []
    eeg_feats_total = []
    fmri_map_feats_total = []
    eeg_map_feats_total = []
    bad_tr_total = []
    eeg_index_total = []
    predictions_total = []
    indices_total = []
    
    net_device = next(model.parameters()).device
    model.eval()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        inputs = {
            "fmri": batch_data_label["fmri"].float(),
            "eeg": batch_data_label["eeg"].float(),
            "bad_tr": batch_data_label["bad_tr"].float(),
            "eeg_index": batch_data_label["eeg_index"].float(),
        }
        outputs = model(inputs, is_train=False)
        input_predictions = outputs["predictions"]["vigilance_head"].squeeze(0)
        fmri_feats_total.append(np.array(outputs["fmri_feats"]))
        eeg_feats_total.append(np.array(outputs["eeg_feats"]))
        fmri_map_feats_total.append(np.array(outputs["fmri_map_feats"]))
        eeg_map_feats_total.append(np.array(outputs["eeg_map_feats"]))
        bad_tr_total.append(np.array(outputs["bad_tr"]))
        eeg_index_total.append(np.array(batch_data_label["eeg_index"]))
        predictions_total.append(np.array(input_predictions))
        indices_total.append(np.array(batch_data_label["indices"]))

    fmri_feats_total = np.concatenate(fmri_feats_total, axis=0)
    eeg_feats_total = np.concatenate(eeg_feats_total, axis=0)
    fmri_map_feats_total = np.concatenate(fmri_map_feats_total, axis=0)
    eeg_map_feats_total = np.concatenate(eeg_map_feats_total, axis=0)
    bad_tr_total = np.concatenate(bad_tr_total, axis=0)
    eeg_index_total = np.concatenate(eeg_index_total, axis=0)
    predictions_total = np.concatenate(predictions_total, axis=0)
    indices_total = np.concatenate(indices_total, axis=0)
    
    print(f"fmri_feats_total: {fmri_feats_total.shape}")
    print(f"eeg_feats_total: {eeg_feats_total.shape}")
    print(f"fmri_map_feats_total: {fmri_map_feats_total.shape}")
    print(f"eeg_map_feats_total: {eeg_map_feats_total.shape}")
    print(f"bad_tr_total: {bad_tr_total.shape}")
    print(f"eeg_index_total: {eeg_index_total.shape}")
    print(f"predictions_total: {predictions_total.shape}")
    print(f"indices_total: {indices_total.shape}")
    fmri_feats_total = fmri_feats_total.reshape(fmri_feats_total.shape[0], -1)
    eeg_feats_total = eeg_feats_total.reshape(eeg_feats_total.shape[0], -1)
    fmri_map_feats_total = fmri_map_feats_total.reshape(fmri_map_feats_total.shape[0], -1)
    eeg_map_feats_total = eeg_map_feats_total.reshape(eeg_map_feats_total.shape[0], -1)
    bad_tr_total = bad_tr_total.reshape(bad_tr_total.shape[0], -1)
    eeg_index_total = eeg_index_total.reshape(eeg_index_total.shape[0], -1)
    predictions_total = predictions_total.reshape(predictions_total.shape[0], -1)
    indices_total = indices_total.reshape(indices_total.shape[0], -1)

    base_folder = args.test_logger
    new_folder = os.path.join(base_folder, "predictions_"+dataset_config.name)
    os.makedirs(new_folder, exist_ok=True)  # Create folder if it doesn't exist

    fmri_feats_total_path = os.path.join(new_folder, "fmri_feats_total.csv")
    np.savetxt(fmri_feats_total_path, fmri_feats_total, delimiter=",", fmt="%.5f")
    print(f"fmri_feats_total saved at: {fmri_feats_total_path}")

    eeg_feats_total_path = os.path.join(new_folder, "eeg_feats_total.csv")
    np.savetxt(eeg_feats_total_path, eeg_feats_total, delimiter=",", fmt="%.5f")
    print(f"eeg_feats_total saved at: {eeg_feats_total_path}")

    fmri_map_feats_total_path = os.path.join(new_folder, "fmri_map_feats_total.csv")
    np.savetxt(fmri_map_feats_total_path, fmri_map_feats_total, delimiter=",", fmt="%.5f")
    print(f"fmri_map_feats_total saved at: {fmri_map_feats_total_path}")

    eeg_map_feats_total_path = os.path.join(new_folder, "eeg_map_feats_total.csv")
    np.savetxt(eeg_map_feats_total_path, eeg_map_feats_total, delimiter=",", fmt="%.5f")
    print(f"eeg_map_feats_total saved at: {eeg_map_feats_total_path}")

    bad_tr_total_path = os.path.join(new_folder, "bad_tr_total.csv")
    np.savetxt(bad_tr_total_path, bad_tr_total, delimiter=",", fmt="%.5f")
    print(f"bad_tr_total saved at: {bad_tr_total_path}")

    eeg_index_total_path = os.path.join(new_folder, "eeg_index_total.csv")
    np.savetxt(eeg_index_total_path, eeg_index_total, delimiter=",", fmt="%.5f")
    print(f"eeg_index_total saved at: {eeg_index_total_path}")

    predictions_total_path = os.path.join(new_folder, "predictions_total.csv")
    np.savetxt(predictions_total_path, predictions_total, delimiter=",", fmt="%.5f")
    print(f"predictions_total saved at: {predictions_total_path}")
    
    indices_total_path = os.path.join(new_folder, "indices_total.csv")
    np.savetxt(indices_total_path, indices_total, delimiter=",", fmt="%.5f")
    print(f"indices_total saved at: {indices_total_path}")

    
@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
):

    acre_calculator = ACRECalculator(
        dataset_config=dataset_config,
    )
    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)
    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    
    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        inputs = {
            "fmri": batch_data_label["fmri"].float(),
            "eeg": batch_data_label["eeg"].float(),
            "bad_tr": batch_data_label["bad_tr"].float(),
            "eeg_index": batch_data_label["eeg_index"].float(),
        }
        outputs = model(inputs, is_train=False)

        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers GT tensor across all ranks
        outputs = all_gather_dict(outputs["predictions"])
        batch_data_label = all_gather_dict(batch_data_label)
        acre_calculator.step_meter(outputs, batch_data_label)
        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg
        curr_iter += 1
        barrier()
    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

        if args.test_visualize:
            visualize(dataset_loader, dataset_config, model, logger, args)

    return acre_calculator
