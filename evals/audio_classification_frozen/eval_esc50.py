# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
ESC-50 Audio Classification Evaluation with Frozen Encoder

This module provides evaluation on ESC-50 using frozen MJEPA encoders
with attentive classifier probes.

Key features:
- 5-fold cross-validation (standard ESC-50 evaluation protocol)
- Single-label classification with CrossEntropyLoss
- Accuracy metric (not mAP like AudioSet)
- Reports mean ± std accuracy across all 5 folds
"""

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from evals.audio_classification_frozen.common import (
    EMAModel,
    gather_tensor,
    init_opt,
    load_checkpoint,
    mixup_data,
    random_time_shift,
    spec_augment,
)
from evals.audio_classification_frozen.models import init_module
from src.datasets.esc50_eval_dataset import ESC50FoldDataset
from src.models.attentive_pooler import AttentiveClassifier, LinearClassifier
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):
    """
    ESC-50 5-fold cross-validation evaluation.

    Trains and evaluates on all 5 folds, reporting average accuracy.
    ESC-50 is single-label classification using CrossEntropyLoss and Accuracy metric.
    """

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- VAL ONLY
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs", {})

    args_exp = args_eval.get("experiment")

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)
    classifier_type = args_classifier.get("classifier_type", "attentive")

    # -- DATA
    args_data = args_exp.get("data")
    num_classes = args_data.get("num_classes", 50)
    data_dir = args_data.get("data_dir")
    labels_csv = args_data.get("labels_csv")
    num_folds = args_data.get("num_folds", 5)

    # Audio-specific params
    num_mel_bins = args_data.get("num_mel_bins", 128)
    target_length = args_data.get("target_length", 512)
    sample_rate = args_data.get("sample_rate", 16000)
    audio_mean = args_data.get("audio_mean", -4.268)
    audio_std = args_data.get("audio_std", 4.569)

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16", False)
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]

    # -- AUGMENTATIONS
    args_aug = args_exp.get("augmentations", {})
    use_mixup = args_aug.get("use_mixup", False)
    mixup_alpha = args_aug.get("mixup_alpha", 0.4)
    use_specaugment = args_aug.get("use_specaugment", False)
    time_mask_param = args_aug.get("time_mask_param", 50)
    freq_mask_param = args_aug.get("freq_mask_param", 10)
    num_time_masks = args_aug.get("num_time_masks", 2)
    num_freq_masks = args_aug.get("num_freq_masks", 2)
    use_time_shift = args_aug.get("use_time_shift", False)
    time_shift_ratio = args_aug.get("time_shift_ratio", 0.2)
    use_ema = args_aug.get("use_ema", False)
    ema_decay = args_aug.get("ema_decay", 0.999)

    aug_config = {
        "use_mixup": use_mixup,
        "mixup_alpha": mixup_alpha,
        "use_specaugment": use_specaugment,
        "time_mask_param": time_mask_param,
        "freq_mask_param": freq_mask_param,
        "num_time_masks": num_time_masks,
        "num_freq_masks": num_freq_masks,
        "use_time_shift": use_time_shift,
        "time_shift_ratio": time_shift_ratio,
        "use_ema": use_ema,
        "ema_decay": ema_decay,
    }

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "esc50_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")

    # -- make csv_logger
    csv_logger = None
    if rank == 0:
        csv_logger = CSVLogger(
            log_file,
            ("%d", "fold"),
            ("%d", "epoch"),
            ("%.5f", "train_loss"),
            ("%.5f", "val_loss"),
            ("%.5f", "accuracy"),
        )

    # -- Log settings
    logger.info("=" * 60)
    logger.info("ESC-50 5-FOLD CROSS-VALIDATION")
    logger.info("=" * 60)
    logger.info(f"  Number of folds:    {num_folds}")
    logger.info(f"  Number of classes:  {num_classes}")
    mixup_str = f"ENABLED (alpha={mixup_alpha})" if use_mixup else "disabled"
    logger.info(f"  Mixup:              {mixup_str}")
    ema_str = f"ENABLED (decay={ema_decay})" if use_ema else "disabled"
    logger.info(f"  EMA:                {ema_str}")
    logger.info("=" * 60)

    # -- init encoder (shared across all folds)
    encoder = init_module(
        module_name=module_name,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    # Create fold dataset manager
    fold_manager = ESC50FoldDataset(
        data_dir=data_dir,
        labels_csv=labels_csv,
        num_mel_bins=num_mel_bins,
        target_length=target_length,
        sample_rate=sample_rate,
        mean=audio_mean,
        std=audio_std,
    )

    # Store best accuracy per fold
    fold_accuracies = []

    # =========================================================================
    # MAIN 5-FOLD LOOP
    # =========================================================================
    for fold in range(1, num_folds + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"FOLD {fold}/{num_folds}")
        logger.info("=" * 60)

        # Get datasets for this fold
        train_dataset, val_dataset = fold_manager.get_fold_datasets(fold)

        # Create dataloaders
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        ipe = len(train_loader)
        ipe_val = len(val_loader)
        logger.info(f"  Train samples: {len(train_dataset)}, iterations/epoch: {ipe}")
        logger.info(f"  Val samples: {len(val_dataset)}, iterations/epoch: {ipe_val}")

        # -- init classifiers for this fold (fresh initialization)
        if fold == 1:
            logger.info(f"Classifier type: {classifier_type}")
        if classifier_type == "attentive":
            classifiers = [
                AttentiveClassifier(
                    embed_dim=encoder.embed_dim,
                    num_heads=num_heads,
                    depth=num_probe_blocks,
                    num_classes=num_classes,
                    use_activation_checkpointing=True,
                ).to(device)
                for _ in opt_kwargs
            ]
        elif classifier_type == "linear":
            classifiers = [
                LinearClassifier(
                    embed_dim=encoder.embed_dim,
                    num_classes=num_classes,
                    pool="mean",
                ).to(device)
                for _ in opt_kwargs
            ]
        else:
            raise ValueError(f"Unsupported classifier_type='{classifier_type}'. " "Use 'attentive' or 'linear'.")
        if world_size > 1:
            classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]

        # -- init optimizer and scheduler for this fold
        optimizer, scaler, scheduler, wd_scheduler = init_opt(
            classifiers=classifiers,
            opt_kwargs=opt_kwargs,
            iterations_per_epoch=ipe,
            num_epochs=num_epochs,
            use_bfloat16=use_bfloat16,
        )

        # -- init EMA for this fold
        ema_models = None
        if use_ema:
            ema_models = [EMAModel(c, decay=ema_decay) for c in classifiers]

        # Checkpoint paths for this fold
        fold_folder = os.path.join(folder, f"fold_{fold}")
        if not os.path.exists(fold_folder):
            os.makedirs(fold_folder, exist_ok=True)
        latest_path = os.path.join(fold_folder, "latest.pt")
        best_path = os.path.join(fold_folder, "best.pt")

        # -- load checkpoint for this fold if exists
        start_epoch = 0
        best_acc = 0.0
        if resume_checkpoint and os.path.exists(latest_path):
            classifiers, optimizer, scaler, start_epoch, best_acc = load_checkpoint(
                r_path=latest_path,
                classifiers=classifiers,
                opt=optimizer,
                scaler=scaler,
                val_only=val_only,
            )
            for _ in range(start_epoch * ipe):
                [s.step() for s in scheduler]
                [wds.step() for wds in wd_scheduler]

        def save_fold_checkpoint(epoch, is_best=False):
            all_classifier_dicts = [c.state_dict() for c in classifiers]
            all_opt_dicts = [o.state_dict() for o in optimizer]
            save_dict = {
                "classifiers": all_classifier_dicts,
                "opt": all_opt_dicts,
                "scaler": None if scaler is None else [s.state_dict() for s in scaler],
                "epoch": epoch,
                "fold": fold,
                "batch_size": batch_size,
                "world_size": world_size,
                "best_mAP": best_acc,  # Using mAP key for compatibility
            }
            if rank == 0:
                torch.save(save_dict, latest_path)
                if is_best:
                    torch.save(save_dict, best_path)

        # TRAIN LOOP for this fold
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"Fold {fold}, Epoch {epoch + 1}/{num_epochs}")
            train_sampler.set_epoch(epoch)

            if val_only:
                train_loss = -1.0
            else:
                train_loss = run_one_epoch(
                    device=device,
                    training=True,
                    encoder=encoder,
                    classifiers=classifiers,
                    scaler=scaler,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    wd_scheduler=wd_scheduler,
                    data_loader=train_loader,
                    use_bfloat16=use_bfloat16,
                    aug_config=aug_config,
                    ema_models=ema_models,
                    num_classes=num_classes,
                )

            # Apply EMA for validation
            if use_ema and ema_models is not None:
                for ema, c in zip(ema_models, classifiers):
                    ema.apply_shadow(c)

            val_results = run_one_epoch(
                device=device,
                training=False,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=val_loader,
                use_bfloat16=use_bfloat16,
                aug_config=aug_config,
                ema_models=None,
            )

            # Restore original weights
            if use_ema and ema_models is not None:
                for ema, c in zip(ema_models, classifiers):
                    ema.restore(c)

            val_loss = val_results["loss"]
            val_acc = val_results["accuracy"]

            logger.info(
                f"  [Fold {fold}][Epoch {epoch + 1}] train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | accuracy: {val_acc * 100:.2f}%"
            )

            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                logger.info(f"  New best accuracy for fold {fold}: {best_acc * 100:.2f}%")

            if rank == 0:
                if csv_logger is not None:
                    csv_logger.log(fold, epoch + 1, train_loss, val_loss, val_acc)

            if val_only:
                break

            save_fold_checkpoint(epoch + 1, is_best=is_best)

        # Record best accuracy for this fold
        fold_accuracies.append(best_acc)
        logger.info(f"Fold {fold} complete. Best accuracy: {best_acc * 100:.2f}%")

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("ESC-50 5-FOLD CROSS-VALIDATION RESULTS")
    logger.info("=" * 60)
    for i, acc in enumerate(fold_accuracies, 1):
        logger.info(f"  Fold {i}: {acc * 100:.2f}%")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    logger.info("-" * 40)
    logger.info(f"  Mean Accuracy: {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")
    logger.info("=" * 60)

    return mean_acc, std_acc


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    aug_config=None,
    ema_models=None,
    num_classes=50,
):
    """
    Run one epoch for ESC-50 (single-label classification).

    Uses CrossEntropyLoss and computes Accuracy.

    Args:
        num_classes: Number of classes for one-hot encoding in mixup.
    """
    if aug_config is None:
        aug_config = {}

    use_mixup = aug_config.get("use_mixup", False) and training
    mixup_alpha = aug_config.get("mixup_alpha", 0.4)
    use_specaugment = aug_config.get("use_specaugment", False) and training
    time_mask_param = aug_config.get("time_mask_param", 50)
    freq_mask_param = aug_config.get("freq_mask_param", 10)
    num_time_masks = aug_config.get("num_time_masks", 2)
    num_freq_masks = aug_config.get("num_freq_masks", 2)
    use_time_shift = aug_config.get("use_time_shift", False) and training
    time_shift_ratio = aug_config.get("time_shift_ratio", 0.2)

    for c in classifiers:
        c.train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    loss_meters = [AverageMeter() for _ in classifiers]

    # For accuracy computation during validation
    all_predictions = [[] for _ in classifiers]
    all_targets = []

    for itr, (av_data, labels) in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            audio_data = av_data["audio"].to(device, non_blocking=True)
            labels = labels.to(device)

            # ============ APPLY AUGMENTATIONS ============
            if training:
                if use_time_shift:
                    audio_data = random_time_shift(audio_data, max_shift_ratio=time_shift_ratio)

                if use_specaugment:
                    audio_data = spec_augment(
                        audio_data,
                        time_mask_param=time_mask_param,
                        freq_mask_param=freq_mask_param,
                        num_time_masks=num_time_masks,
                        num_freq_masks=num_freq_masks,
                    )

                # Mixup for single-label: creates soft labels
                if use_mixup:
                    # Convert labels to one-hot for mixup (num_classes passed as parameter)
                    labels_onehot = torch.zeros(labels.size(0), num_classes, device=device)
                    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                    audio_data, labels_soft, _ = mixup_data(audio_data, labels_onehot, alpha=mixup_alpha)
                    use_soft_labels = True
                else:
                    labels_soft = None
                    use_soft_labels = False
            else:
                use_soft_labels = False
                labels_soft = None
            # ==============================================

            with torch.no_grad():
                outputs = encoder(audio_data)

            if training:
                classifier_outputs = [c(outputs) for c in classifiers]
            else:
                with torch.no_grad():
                    classifier_outputs = [c(outputs) for c in classifiers]

        # Compute losses
        if use_soft_labels:
            # Cross entropy with soft labels
            losses = [torch.nn.functional.cross_entropy(co, labels_soft) for co in classifier_outputs]
        else:
            losses = [criterion(co, labels) for co in classifier_outputs]

        for lm, loss in zip(loss_meters, losses):
            lm.update(loss.item())

        # Collect predictions for accuracy (validation only)
        if not training:
            with torch.no_grad():
                for i, co in enumerate(classifier_outputs):
                    preds = torch.argmax(co, dim=1)
                    all_predictions[i].append(preds.cpu())
                all_targets.append(labels.cpu())

        if training:
            if use_bfloat16:
                for s, lo in zip(scaler, losses):
                    s.scale(lo).backward()
                for s, c, o in zip(scaler, classifiers, optimizer):
                    s.unscale_(o)
                    torch.nn.utils.clip_grad_norm_(c.parameters(), max_norm=1.0)
                    s.step(o)
                    s.update()
            else:
                for lo in losses:
                    lo.backward()
                for c, o in zip(classifiers, optimizer):
                    torch.nn.utils.clip_grad_norm_(c.parameters(), max_norm=1.0)
                    o.step()
            for o in optimizer:
                o.zero_grad()

            if ema_models is not None:
                for ema, c in zip(ema_models, classifiers):
                    ema.update(c)

        if itr % 10 == 0:
            _agg_loss = np.array([lm.avg for lm in loss_meters])
            logger.info(
                "[%5d] loss: %.4f [%.4f - %.4f] [mem: %.2e]"
                % (
                    itr,
                    _agg_loss.min(),
                    _agg_loss.mean(),
                    _agg_loss.max(),
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
            )

    if training:
        return np.array([lm.avg for lm in loss_meters]).min()
    else:
        # Compute accuracy
        all_targets = torch.cat(all_targets, dim=0)

        best_acc = 0.0
        for i, preds_list in enumerate(all_predictions):
            preds = torch.cat(preds_list, dim=0)

            # Gather across processes
            preds_gathered = gather_tensor(preds)
            targets_gathered = gather_tensor(all_targets)

            if preds_gathered is not None:
                correct = (preds_gathered == targets_gathered).sum().item()
                total = targets_gathered.size(0)
                acc = correct / total
                if acc > best_acc:
                    best_acc = acc

        return {
            "loss": np.array([lm.avg for lm in loss_meters]).min(),
            "accuracy": best_acc,
        }


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

# ESC-50 with 5-fold cross-validation
"""
python -m evals.audio_classification_frozen.eval \
    --config configs/eval/esc50/esc50_augmented.yaml
"""
