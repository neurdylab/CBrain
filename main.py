# Copyright (c) Facebook, Inc. and its affiliates.
# This codebase is built upon 3DETR and OV-3DET

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import random
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler
from datasets_shuffle import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from optimizer import build_optimizer
from criterion import build_criterion
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier
from utils.misc import my_worker_init_fn
from utils.io import save_checkpoint, resume_if_possible
from utils.logger import Logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def make_args_parser():
    parser = argparse.ArgumentParser("CBrain", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="cbrain",
        type=str,
        help="Name of the model",
        choices=["cbrain",
                 "cbrain_purefmri",
                 "random_guess",
                 "svm_hierarchy"],
    )
    
    parser.add_argument("--fmri_sp_enc_nlayers", default=3, type=int)
    parser.add_argument("--fmri_sp_enc_dim", default=66, type=int)
    parser.add_argument("--fmri_sp_enc_ffn_dim", default=128, type=int)
    parser.add_argument("--fmri_sp_enc_dropout", default=0.1, type=float)
    parser.add_argument("--fmri_sp_enc_nhead", default=3, type=int)
    parser.add_argument("--fmri_sp_enc_pos_embed", default=None, type=str)
    parser.add_argument("--fmri_sp_enc_activation", default="relu", type=str)

    parser.add_argument("--fmri_te_enc_nlayers", default=3, type=int)
    parser.add_argument("--fmri_te_enc_dim", default=10, type=int)
    parser.add_argument("--fmri_te_enc_ffn_dim", default=128, type=int)
    parser.add_argument("--fmri_te_enc_dropout", default=0.1, type=float)
    parser.add_argument("--fmri_te_enc_nhead", default=1, type=int)
    parser.add_argument("--fmri_te_enc_pos_embed", default=None, type=str)
    parser.add_argument("--fmri_te_enc_activation", default="relu", type=str)

    parser.add_argument("--eeg_sp_enc_nlayers", default=3, type=int)
    parser.add_argument("--eeg_sp_enc_dim", default=26, type=int)
    parser.add_argument("--eeg_sp_enc_ffn_dim", default=128, type=int)
    parser.add_argument("--eeg_sp_enc_dropout", default=0.1, type=float)
    parser.add_argument("--eeg_sp_enc_nhead", default=2, type=int)
    parser.add_argument("--eeg_sp_enc_pos_embed", default=None, type=str)
    parser.add_argument("--eeg_sp_enc_activation", default="relu", type=str)

    parser.add_argument("--eeg_enc_nlayers", default=3, type=int)
    parser.add_argument("--eeg_enc_dim", default=10, type=int)
    parser.add_argument("--eeg_enc_ffn_dim", default=128, type=int)
    parser.add_argument("--eeg_enc_dropout", default=0.1, type=float)
    parser.add_argument("--eeg_enc_nhead", default=1, type=int)
    parser.add_argument("--eeg_enc_pos_embed", default=None, type=str)
    parser.add_argument("--eeg_enc_activation", default="relu", type=str)

    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=2,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ##### Set Loss #####
    parser.add_argument("--l2_loss", default=False, action="store_true")
    parser.add_argument("--loss_fmri_prediction_weight", default=0.1, type=float)
    parser.add_argument("--loss_fmri_label_weight", default=0, type=float)
    parser.add_argument("--loss_eeg_label_weight", default=0, type=float)
    parser.add_argument("--loss_fmri_eeg_weight", default=0, type=float)

    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", required=True, type=str, choices=["eegfmri_vu",
                                                            "NIH_ecr",
                                                            "NIH_ect"]
    )

    parser.add_argument("--dataset_num_workers", default=1, type=int)
    parser.add_argument("--batchsize_per_gpu", default=4, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=10, type=int)
    parser.add_argument("--eval_every_epoch", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--train_test", default=False, action="store_true")
    parser.add_argument("--test_visualize", default=False, action="store_true")
    parser.add_argument("--smooth_visualize", default=False, action="store_true")
    parser.add_argument("--test_logger", default=None, type=str)
    parser.add_argument("--test_ckpt", default=None, type=str)

    ##### Zero-shot Testing on New Datasets #####
    parser.add_argument("--zero_shot", default=False, action="store_true")

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=5, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=50, type=int)

    ##### Distributed Training #####
    parser.add_argument("--train_device", default="cpu", type=str)
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

    return parser


def do_train(
    args,
    model,
    model_no_ddp,
    optimizer,
    criterion,
    dataset_config,
    dataloaders,
    best_val_metrics,
):
    """
    Main training loop.
    This trains the model for `args.max_epoch` epochs and tests the model after every `args.eval_every_epoch`.
    We always evaluate the final checkpoint.
    """

    num_iters_per_epoch = len(dataloaders["train"])
    num_iters_per_eval_epoch = len(dataloaders["test"])
    print(f"Model is {model}")
    print(f"Training started at epoch {args.start_epoch} until {args.max_epoch}.")
    print(f"One training epoch = {num_iters_per_epoch} iters.")
    print(f"One eval epoch = {num_iters_per_eval_epoch} iters.")

    final_eval = os.path.join(args.checkpoint_dir, "final_eval.txt")
    final_eval_pkl = os.path.join(args.checkpoint_dir, "final_eval.pkl")

    if os.path.isfile(final_eval):
        print(f"Found final eval file {final_eval}. Skipping training.")
        return

    logger = Logger(args.checkpoint_dir)

    for epoch in range(args.start_epoch, args.max_epoch):
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(epoch)

        acs = train_one_epoch(
            args,
            epoch,
            model,
            optimizer,
            criterion,
            dataset_config,
            dataloaders["train"],
            logger,
        )

        # latest checkpoint is always stored in checkpoint.pth
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )

        metrics = acs.compute_metrics()
        metric_str = acs.metrics_to_str(metrics, per_class=False)
        metrics_dict = acs.metrics_to_dict(metrics)
        curr_iter = epoch * len(dataloaders["train"])
        if is_primary():
            print("==" * 10)
            print(f"Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
            print("==" * 10)
            logger.log_scalars(metrics_dict, curr_iter, prefix="Train/")

        if (
            epoch > 0
            and args.save_separate_checkpoint_every_epoch > 0
            and epoch % args.save_separate_checkpoint_every_epoch == 0
        ):
            # separate checkpoints are stored as checkpoint_{epoch}.pth
            save_checkpoint(
                args.checkpoint_dir,
                model_no_ddp,
                optimizer,
                epoch,
                args,
                best_val_metrics,
            )

        if epoch % args.eval_every_epoch == 0 or epoch == (args.max_epoch - 1):
            ac_calculator = evaluate(
                args,
                epoch,
                model,
                criterion,
                dataset_config,
                dataloaders["test"],
                logger,
                curr_iter,
            )
            metrics = ac_calculator.compute_metrics()
            f1 = metrics["mF1"]
            metric_str = ac_calculator.metrics_to_str(metrics, per_class=True)
            metrics_dict = ac_calculator.metrics_to_dict(metrics)
            if is_primary():
                print("==" * 10)
                print(f"Evaluate Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
                print("==" * 10)
                logger.log_scalars(metrics_dict, curr_iter, prefix="Test/")

            if is_primary() and (
                len(best_val_metrics) == 0 or best_val_metrics["mF1"] < f1
            ):
                best_val_metrics = metrics
                filename = "checkpoint_best.pth"
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    epoch,
                    args,
                    best_val_metrics,
                    filename=filename,
                )
                print(
                    f"Epoch [{epoch}/{args.max_epoch}] saved current best val checkpoint at {filename}; mF1 {f1}"
                )

    # always evaluate last checkpoint
    epoch = args.max_epoch - 1
    curr_iter = epoch * len(dataloaders["train"])
    ac_calculator = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
    )
    metrics = ac_calculator.compute_metrics()
    metric_str = ac_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(f"Evaluate Final [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
        print("==" * 10)

        with open(final_eval, "w") as fh:
            fh.write("Training Finished.\n")
            fh.write("==" * 10)
            fh.write("Final Eval Numbers.\n")
            fh.write(metric_str)
            fh.write("\n")
            fh.write("==" * 10)
            fh.write("Best Eval Numbers.\n")
            fh.write(ac_calculator.metrics_to_str(best_val_metrics))
            fh.write("\n")

        with open(final_eval_pkl, "wb") as fh:
            pickle.dump(metrics, fh)

def test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)
    
    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"])
    logger = Logger(args.test_logger)
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    ac_calculator = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
    )
    metrics = ac_calculator.compute_metrics()
    metric_str = ac_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(f"Test model; Metrics {metric_str}")
        print("==" * 10)


def train_test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)
    
    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"])
    logger = Logger(args.test_logger)
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    ac_calculator = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["train"],
        logger,
        curr_iter,
    )
    metrics = ac_calculator.compute_metrics()
    metric_str = ac_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(f"Train Test model; Metrics {metric_str}")
        print("==" * 10)


def zero_shot_test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)
    
    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"])
    logger = Logger(args.test_logger)
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    ac_calculator = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["zero_shot"],
        logger,
        curr_iter,
    )
    metrics = ac_calculator.compute_metrics()
    metric_str = ac_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(f"Zero Shot Test model; Metrics {metric_str}")
        print("==" * 10)


def main(local_rank, args):
    if args.ngpus > 1:
        print(
            "Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. Use at your own risk :)"
        )
        print("To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch to greater than 50")
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    print(f"Called with args: {args}")
    if args.train_device == "cpu":
        device = torch.device("cpu")
        torch.set_default_device(device)
        np.random.seed(args.seed)
        random.seed(args.seed)
    else:
        torch.cuda.set_device(local_rank)
        np.random.seed(args.seed + get_rank())
        torch.manual_seed(args.seed + get_rank())
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + get_rank())
        torch.backends.cudnn.deterministic = True  # Forces deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disables auto-tuning for convolutions
        os.environ["PYTHONHASHSEED"] = str(args.seed)  # Ensures hash-based ops remain deterministic


    datasets, dataset_config = build_dataset(args)
    model = build_model(args, dataset_config)

    if args.train_device == "cpu":
        model = model.to(device)
    else:
        model = model.cuda(local_rank)

    model_no_ddp = model

    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    criterion = build_criterion(args, dataset_config)

    if args.train_device == "cpu":
        criterion = criterion.to(device)
    else:
        criterion = criterion.cuda(local_rank)

    dataloaders = {}
    if args.test_only:
        dataset_splits = ["test"]
    elif args.train_test:
        dataset_splits = ["train"]
    elif args.zero_shot:
        dataset_splits = ["zero_shot"]
    else:
        dataset_splits = ["train", "test"]
    for split in dataset_splits:
        if split == "train":
            shuffle = True
        else:
            shuffle = False
        if is_distributed():
            sampler = DistributedSampler(datasets[split], shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
        )
        dataloaders[split + "_sampler"] = sampler

    if args.test_only:
        criterion = None  # faster evaluation
        test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders)
    elif args.train_test:
        criterion = None  # faster evaluation
        train_test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders)
    elif args.zero_shot:
        criterion = None
        zero_shot_test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders)
    else:
        assert (
            args.checkpoint_dir is not None
        ), f"Please specify a checkpoint dir using --checkpoint_dir"
        if is_primary() and not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        optimizer = build_optimizer(args, model_no_ddp)
        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer
        )
        args.start_epoch = loaded_epoch + 1
        do_train(
            args,
            model,
            model_no_ddp,
            optimizer,
            criterion,
            dataset_config,
            dataloaders,
            best_val_metrics,
        )

def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)
