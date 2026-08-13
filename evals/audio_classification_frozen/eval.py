# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
AudioSet/FSD50K Audio Classification Evaluation with Frozen Encoder

This module provides evaluation for multi-label audio classification
on AudioSet (20K balanced) and FSD50K using frozen MJEPA encoders
with attentive classifier probes.

Key features:
- Multiple classifier heads with different hyperparameters (learning rates, weight decays)
- Multi-label classification with BCEWithLogitsLoss and mAP metric
- Distributed training with native PyTorch DDP
- WarmupCosine LR scheduler and Cosine weight decay scheduler
- Resume from checkpoint support
- CSV logging
For ESC-50 evaluation (single-label, 5-fold CV), see eval_esc50.py
"""

import os
import warnings

# Suppress torch deprecation warnings that don't affect functionality
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.backends.cuda.sdp_kernel.*")

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

# Import shared utilities from common.py
from evals.audio_classification_frozen.common import (
    EMAModel,
    compute_mAP,
    gather_tensor,
    init_opt,
    load_checkpoint,
    mixup_data,
    random_time_shift,
    spec_augment,
    volume_jitter,
)
from evals.audio_classification_frozen.models import init_module
from src.datasets.fsd50k_eval_dataset import FSD50KEval, FSD50KMultiClipCollator
from src.datasets.legacy_audioset_eval_dataset_audio_only import AudioSetAudioEval
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
    Main function for AudioSet/FSD50K multi-label classification evaluation.

    For ESC-50 (dataset_type='esc50'), this dispatches to eval_esc50.main().
    """
    # Check dataset type and dispatch to specialized handler
    dataset_type = args_eval.get("experiment", {}).get("data", {}).get("dataset_type", "audioset")
    if dataset_type == "esc50":
        from evals.audio_classification_frozen.eval_esc50 import main as main_esc50

        return main_esc50(args_eval=args_eval, resume_preempt=resume_preempt)
    elif dataset_type in ("audioset_video", "audioset_audio_video"):
        from evals.audio_classification_frozen.eval_video import main as main_video

        return main_video(args_eval=args_eval, resume_preempt=resume_preempt)

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
    num_classes = args_data.get("num_classes", 527)
    train_audio_dir = args_data.get("train_audio_dir")
    train_csv = args_data.get("train_csv")
    val_audio_dir = args_data.get("val_audio_dir")
    val_csv = args_data.get("val_csv")

    # Audio-specific params
    num_mel_bins = args_data.get("num_mel_bins", 128)
    target_length = args_data.get("target_length", 1024)
    sample_rate = args_data.get("sample_rate", 16000)
    audio_mean = args_data.get("audio_mean", -4.268)
    audio_std = args_data.get("audio_std", 4.569)
    target_length_seconds = args_data.get("target_length_seconds", 10.0)  # For waveform datasets

    # Dataset-specific params
    dataset_type = args_data.get("dataset_type", "audioset")
    vocabulary_csv = args_data.get("vocabulary_csv", None)
    num_clips_train = args_data.get("num_clips_train", 1)
    num_clips_eval = args_data.get("num_clips_eval", 1)

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

    # -- AUGMENTATIONS (all disabled by default for backward compatibility)
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
    use_volume_jitter = args_aug.get("use_volume_jitter", False)
    volume_jitter_range = args_aug.get("volume_jitter_range", 0.2)
    use_ema = args_aug.get("use_ema", False)
    ema_decay = args_aug.get("ema_decay", 0.999)
    # Augmentation scheduling: gradually ramp up augmentation strength
    # over the first N epochs (0 = no warmup, use full strength immediately)
    aug_warmup_epochs = args_aug.get("aug_warmup_epochs", 0)

    # Package augmentation config for passing to training
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
        "use_volume_jitter": use_volume_jitter,
        "volume_jitter_range": volume_jitter_range,
        "use_ema": use_ema,
        "ema_decay": ema_decay,
        "aug_warmup_epochs": aug_warmup_epochs,
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
    folder = os.path.join(pretrain_folder, "audio_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    best_path = os.path.join(folder, "best.pt")

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%.5f", "train_loss"),
            ("%.5f", "val_loss"),
            ("%.5f", "mAP"),
            ("%.5f", "mAUC"),
        )

    # -- Log augmentation settings
    logger.info("=" * 60)
    logger.info("AUGMENTATION SETTINGS")
    logger.info("=" * 60)
    mixup_str = f"ENABLED (alpha={mixup_alpha})" if use_mixup else "disabled"
    logger.info(f"  Mixup:        {mixup_str}")
    specaug_str = f"ENABLED (T={time_mask_param}, F={freq_mask_param})" if use_specaugment else "disabled"
    logger.info(f"  SpecAugment:  {specaug_str}")
    shift_str = f"ENABLED (ratio={time_shift_ratio})" if use_time_shift else "disabled"
    logger.info(f"  Time Shift:   {shift_str}")
    ema_str = f"ENABLED (decay={ema_decay})" if use_ema else "disabled"
    logger.info(f"  EMA:          {ema_str}")
    logger.info("=" * 60)

    # -- init encoder
    encoder = init_module(
        module_name=module_name,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    # -- init classifiers (multiple heads with different hyperparameters)
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
    # Only wrap with DDP if distributed is initialized (world_size > 1)
    if world_size > 1:
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
    logger.info(f"Initialized {len(classifiers)} classifier heads")
    print(classifiers[0])

    # -- init dataloaders
    train_loader, train_sampler = make_dataloader(
        audio_dir=train_audio_dir,
        csv_path=train_csv,
        num_mel_bins=num_mel_bins,
        target_length=target_length,
        sample_rate=sample_rate,
        audio_mean=audio_mean,
        audio_std=audio_std,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        dataset_type=dataset_type,
        vocabulary_csv=vocabulary_csv,
        num_clips=num_clips_train,
        target_length_seconds=target_length_seconds,
    )
    val_loader, _ = make_dataloader(
        audio_dir=val_audio_dir,
        csv_path=val_csv,
        num_mel_bins=num_mel_bins,
        target_length=target_length,
        sample_rate=sample_rate,
        audio_mean=audio_mean,
        audio_std=audio_std,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        dataset_type=dataset_type,
        vocabulary_csv=vocabulary_csv,
        num_clips=num_clips_eval,
        target_length_seconds=target_length_seconds,
    )
    ipe = len(train_loader)
    ipe_val = len(val_loader)

    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Train iterations per epoch: {ipe}")
    logger.info(f"  Val iterations per epoch:   {ipe_val}")
    logger.info(f"  Total epochs:               {num_epochs}")
    logger.info(f"  Total train iterations:     {ipe * num_epochs}")
    logger.info(f"  Batch size per GPU:         {batch_size}")
    logger.info(f"  World size (num GPUs):      {world_size}")
    logger.info(f"  Effective batch size:       {batch_size * world_size}")
    logger.info(f"  Num classifier heads:       {len(opt_kwargs)}")
    logger.info(f"  Num classes:                {num_classes}")
    logger.info("=" * 60)

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # -- load training checkpoint
    start_epoch = 0
    best_mAP = 0.0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch, best_mAP = load_checkpoint(
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch, is_best=False):
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "best_mAP": best_mAP,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if is_best:
                torch.save(save_dict, best_path)

    # -- init EMA model for classifiers if enabled
    ema_models = None
    if use_ema:
        ema_models = [EMAModel(c, decay=ema_decay) for c in classifiers]
        logger.info(f"Initialized EMA for {len(classifiers)} classifier heads " f"with decay={ema_decay}")

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
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
                current_epoch=epoch,
            )

        # For validation with EMA: temporarily apply EMA weights
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
            ema_models=None,  # Don't update EMA during validation
        )

        # Restore original weights after validation
        if use_ema and ema_models is not None:
            for ema, c in zip(ema_models, classifiers):
                ema.restore(c)

        val_loss = val_results["loss"]
        val_mAP = val_results["mAP"]
        val_mAUC = val_results["mAUC"]

        logger.info(
            "[%5d] train_loss: %.4f | val_loss: %.4f | mAP: %.3f%% | mAUC: %.3f%%"
            % (epoch + 1, train_loss, val_loss, val_mAP * 100, val_mAUC * 100)
        )

        is_best = val_mAP > best_mAP
        if is_best:
            best_mAP = val_mAP
            logger.info(f"New best mAP: {best_mAP * 100:.3f}%")

        if rank == 0:
            csv_logger.log(epoch + 1, train_loss, val_loss, val_mAP, val_mAUC)

        if val_only:
            return

        save_checkpoint(epoch + 1, is_best=is_best)

    logger.info(f"Training complete. Best mAP: {best_mAP * 100:.3f}%")


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
    current_epoch=0,
):
    """Run one epoch of training or validation for multi-label classification."""
    # Default augmentation config (all disabled)
    if aug_config is None:
        aug_config = {}

    # Get augmentation warmup schedule
    aug_warmup_epochs = aug_config.get("aug_warmup_epochs", 0)

    # Calculate augmentation scale factor (0 -> 1 over warmup epochs)
    # After warmup_epochs, scale = 1.0 (full augmentation strength)
    if aug_warmup_epochs > 0 and training:
        # Use (epoch + 1) so first epoch has some augmentation
        aug_scale = min(1.0, (current_epoch + 1) / aug_warmup_epochs)
    else:
        aug_scale = 1.0

    use_mixup = aug_config.get("use_mixup", False) and training
    mixup_alpha = aug_config.get("mixup_alpha", 0.4) * aug_scale
    use_specaugment = aug_config.get("use_specaugment", False) and training
    time_mask_param = int(aug_config.get("time_mask_param", 50) * aug_scale)
    freq_mask_param = int(aug_config.get("freq_mask_param", 10) * aug_scale)
    num_time_masks = aug_config.get("num_time_masks", 2)
    num_freq_masks = aug_config.get("num_freq_masks", 2)
    use_time_shift = aug_config.get("use_time_shift", False) and training
    time_shift_ratio = aug_config.get("time_shift_ratio", 0.2) * aug_scale
    use_volume_jitter = aug_config.get("use_volume_jitter", False) and training
    volume_jitter_range = aug_config.get("volume_jitter_range", 0.2) * aug_scale

    # Log augmentation schedule on first iteration of training
    if training and aug_warmup_epochs > 0:
        logger.info(
            f"[Epoch {current_epoch + 1}] Aug scale: {aug_scale:.2f} | "
            f"mixup_alpha: {mixup_alpha:.3f} | time_mask: {time_mask_param} | "
            f"freq_mask: {freq_mask_param} | vol_jitter: {volume_jitter_range:.3f}"
        )

    for c in classifiers:
        c.train(mode=training)

    criterion = torch.nn.BCEWithLogitsLoss()
    loss_meters = [AverageMeter() for _ in classifiers]

    # For mAP computation during validation
    all_predictions = [[] for _ in classifiers]
    all_targets = []

    for itr, (av_data, labels) in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            # Load data and put on GPU
            audio_data = av_data["audio"].to(device, non_blocking=True)
            labels = labels.to(device)

            # Handle multi-clip data from FSD50K
            # Spectrogram: [B, num_clips, C, T, F] = 5D
            # Waveform: [B, num_clips, C, num_samples] = 4D
            is_multiclip = audio_data.dim() >= 4 and audio_data.shape[1] > 1
            is_waveform = audio_data.dim() == 4 and audio_data.shape[2] == 1  # [B, clips, 1, samples]
            num_clips = audio_data.shape[1] if is_multiclip else 1

            if is_multiclip:
                if audio_data.dim() == 5:
                    # Spectrogram: [B, num_clips, C, T, F] -> [B*num_clips, C, T, F]
                    B, num_clips, C, T, F = audio_data.shape
                    audio_data = audio_data.view(B * num_clips, C, T, F)
                elif audio_data.dim() == 4 and is_waveform:
                    # Waveform: [B, num_clips, 1, samples] -> [B*num_clips, 1, samples]
                    B, num_clips, C, S = audio_data.shape
                    audio_data = audio_data.view(B * num_clips, C, S)

            # ============ APPLY AUGMENTATIONS (training only) ============
            if training:
                if use_volume_jitter:
                    audio_data = volume_jitter(audio_data, jitter_range=volume_jitter_range)

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

                if use_mixup:
                    if is_multiclip:
                        labels_expanded = labels.unsqueeze(1).expand(-1, num_clips, -1).reshape(B * num_clips, -1)
                        audio_data, labels_for_loss, _ = mixup_data(audio_data, labels_expanded, alpha=mixup_alpha)
                    else:
                        audio_data, labels_for_loss, _ = mixup_data(audio_data, labels, alpha=mixup_alpha)
                else:
                    if is_multiclip:
                        labels_for_loss = labels.unsqueeze(1).expand(-1, num_clips, -1).reshape(B * num_clips, -1)
                    else:
                        labels_for_loss = labels
            else:
                if is_multiclip:
                    labels_for_loss = labels.unsqueeze(1).expand(-1, num_clips, -1).reshape(B * num_clips, -1)
                else:
                    labels_for_loss = labels
            # =============================================================

            # Forward through encoder
            with torch.no_grad():
                outputs = encoder(audio_data)

            # Forward through classifiers
            if training:
                classifier_outputs = [c(outputs) for c in classifiers]
            else:
                with torch.no_grad():
                    classifier_outputs = [c(outputs) for c in classifiers]

            # For multi-clip during evaluation, aggregate predictions by averaging
            if is_multiclip and not training:
                classifier_outputs_agg = []
                for co in classifier_outputs:
                    co_reshaped = co.view(B, num_clips, -1)
                    co_agg = co_reshaped.mean(dim=1)
                    classifier_outputs_agg.append(co_agg)
                classifier_outputs = classifier_outputs_agg

        # Compute losses
        if training and is_multiclip:
            losses = [criterion(co, labels_for_loss) for co in classifier_outputs]
        else:
            losses = [criterion(co, labels) for co in classifier_outputs]

        # Update loss meters
        for lm, loss in zip(loss_meters, losses):
            lm.update(loss.item())

        # For validation, collect predictions for mAP
        if not training:
            with torch.no_grad():
                for i, co in enumerate(classifier_outputs):
                    preds = torch.sigmoid(co)
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

            # Update EMA after each optimization step
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
        # Compute mAP for validation
        all_targets = torch.cat(all_targets, dim=0)

        best_mAP = 0.0
        best_mAUC = 0.0
        for i, preds_list in enumerate(all_predictions):
            preds = torch.cat(preds_list, dim=0)

            # Gather predictions and targets across all processes
            preds_gathered = gather_tensor(preds)
            targets_gathered = gather_tensor(all_targets)

            if preds_gathered is not None:
                mAP, mAUC = compute_mAP(preds_gathered.float().numpy(), targets_gathered.float().numpy())
                if mAP > best_mAP:
                    best_mAP = mAP
                    best_mAUC = mAUC

        return {
            "loss": np.array([lm.avg for lm in loss_meters]).min(),
            "mAP": best_mAP,
            "mAUC": best_mAUC,
        }


def make_dataloader(
    audio_dir,
    csv_path,
    batch_size,
    world_size,
    rank,
    num_mel_bins=128,
    target_length=1024,
    sample_rate=16000,
    audio_mean=-4.268,
    audio_std=4.569,
    training=False,
    num_workers=12,
    dataset_type="audioset",
    vocabulary_csv=None,
    num_clips=1,
    target_length_seconds=10.0,  # For waveform datasets
):
    """
    Create dataloader for AudioSet or FSD50K.

    Args:
        audio_dir: Directory containing audio files
        csv_path: Path to CSV file with audio metadata
        batch_size: Batch size per GPU
        world_size: Number of distributed processes
        rank: Current process rank
        num_mel_bins: Number of mel filterbank bins
        target_length: Target length of mel spectrogram in frames
        sample_rate: Audio sample rate
        audio_mean: Normalization mean
        audio_std: Normalization std
        training: Whether in training mode
        num_workers: Number of dataloader workers
        dataset_type: Type of dataset ("audioset" or "fsd50k")
        vocabulary_csv: Path to vocabulary CSV (required for FSD50K)
        num_clips: Number of clips to extract per sample (FSD50K only)

    Returns:
        (data_loader, sampler) tuple
    """
    # Validate required paths
    if audio_dir is None:
        raise ValueError(
            f"audio_dir is required for {dataset_type} dataset. "
            "Check your config file for train_audio_dir/val_audio_dir."
        )
    if csv_path is None:
        raise ValueError(
            f"csv_path is required for {dataset_type} dataset. " "Check your config file for train_csv/val_csv."
        )

    logger.info(f"Creating {'train' if training else 'val'} dataloader for {dataset_type}")
    logger.info(f"  audio_dir: {audio_dir}")
    logger.info(f"  csv_path: {csv_path}")

    if dataset_type == "fsd50k":
        if vocabulary_csv is None:
            raise ValueError(
                "vocabulary_csv is required for FSD50K dataset. " "Add 'vocabulary_csv' to your config's data section."
            )

        dataset = FSD50KEval(
            audio_data_dir=audio_dir,
            dataset_csv=csv_path,
            vocabulary_csv=vocabulary_csv,
            num_mel_bins=num_mel_bins,
            target_length=target_length,
            sample_rate=sample_rate,
            mean=audio_mean,
            std=audio_std,
            num_clips=num_clips,
            training=training,
        )

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
        )

        collator = FSD50KMultiClipCollator(flatten_clips=False)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=training,
            collate_fn=collator,
        )

    elif dataset_type == "audioset":
        dataset = AudioSetAudioEval(
            audio_data_dir=audio_dir,
            dataset_csv=csv_path,
            num_mel_bins=num_mel_bins,
            target_length=target_length,
            sample_rate=sample_rate,
            mean=audio_mean,
            std=audio_std,
            audio_only=True,
        )

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=training,
        )

    else:
        raise ValueError(f"Unsupported dataset_type: '{dataset_type}'. " "Supported types: 'audioset', 'fsd50k'. ")

    logger.info(f"  Dataset size: {len(dataset)} samples")

    return data_loader, sampler


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Audio Classification Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--val_only", action="store_true", help="Only run validation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override num epochs")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.val_only:
        config["val_only"] = True
    if args.checkpoint:
        config["model_kwargs"]["checkpoint"] = args.checkpoint
    if args.batch_size:
        config["experiment"]["optimization"]["batch_size"] = args.batch_size
    if args.num_epochs:
        config["experiment"]["optimization"]["num_epochs"] = args.num_epochs

    main(args_eval=config, resume_preempt=False)
