# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
AudioSet Video / Audio-Video Classification Evaluation with Frozen Encoder

This module provides evaluation for multi-label classification on AudioSet (20K balanced)
using frozen MJEPA encoders with video-only or audio-video features, evaluated via
attentive classifier probes.

Supports two modes (controlled by dataset_type in config):
  - "audioset_video": Uses only video features from MJEPA encoder
  - "audioset_audio_video": Uses concatenated audio + video features from MJEPA encoder

Key features:
- Multiple classifier heads with different hyperparameters (learning rates, weight decays)
- Multi-label classification with BCEWithLogitsLoss and mAP metric
- Distributed training with native PyTorch DDP
- Video transforms: random resized crop + horizontal flip (train), center crop (val)
- WarmupCosine LR scheduler and Cosine weight decay scheduler
- Resume from checkpoint support
- CSV logging
"""

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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

# Video transforms
from app.mjepa.video_transforms import make_transforms
from evals.audio_classification_frozen.common import (
    EMAModel,
    compute_mAP,
    gather_tensor,
    init_opt,
    load_checkpoint,
    mixup_data,
)
from evals.audio_classification_frozen.models import init_module
from src.datasets.audioset_eval_dataset import AudioSetEval
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
    Main function for AudioSet video / audio-video classification evaluation.

    Args:
        args_eval: Configuration dict from YAML config
        resume_preempt: Whether to resume from preempted checkpoint
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
    num_classes = args_data.get("num_classes", 527)
    dataset_type = args_data.get("dataset_type", "audioset_video")
    train_csv = args_data.get("train_csv")
    val_csv = args_data.get("val_csv")

    # Video params
    frames_per_clip = args_data.get("frames_per_clip", 16)
    fps = args_data.get("fps", 4)
    crop_size = args_data.get("crop_size", 224)

    # Audio params (needed for audio_video mode)
    num_mel_bins = args_data.get("num_mel_bins", 128)
    target_length = args_data.get("target_length", 1024)
    sample_rate = args_data.get("sample_rate", 16000)
    audio_mean = args_data.get("audio_mean", -4.268)
    audio_std = args_data.get("audio_std", 4.569)

    # Determine modality mode
    use_video = dataset_type in ("audioset_video", "audioset_audio_video")
    use_audio = dataset_type == "audioset_audio_video"

    if not use_video:
        raise ValueError(
            f"eval_video.py only supports 'audioset_video' and 'audioset_audio_video' dataset_type, "
            f"got '{dataset_type}'. For audio-only eval, use eval.py."
        )

    logger.info(f"Evaluation mode: {'audio+video' if use_audio else 'video-only'}")

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
    use_ema = args_aug.get("use_ema", False)
    ema_decay = args_aug.get("ema_decay", 0.999)

    aug_config = {
        "use_mixup": use_mixup,
        "mixup_alpha": mixup_alpha,
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

    # -- Log settings
    logger.info("=" * 60)
    logger.info("EVALUATION SETTINGS")
    logger.info("=" * 60)
    logger.info(f"  Mode:         {'audio+video' if use_audio else 'video-only'}")
    mixup_str = f"ENABLED (alpha={mixup_alpha})" if use_mixup else "disabled"
    logger.info(f"  Mixup:        {mixup_str}")
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
    if world_size > 1:
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
    logger.info(f"Initialized {len(classifiers)} classifier heads")
    print(classifiers[0])

    # -- init dataloaders
    train_loader, train_sampler = make_dataloader(
        csv_path=train_csv,
        frames_per_clip=frames_per_clip,
        fps=fps,
        crop_size=crop_size,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        num_mel_bins=num_mel_bins,
        target_length=target_length,
        audio_sample_rate=sample_rate,
        audio_mean=audio_mean,
        audio_std=audio_std,
    )
    val_loader, _ = make_dataloader(
        csv_path=val_csv,
        frames_per_clip=frames_per_clip,
        fps=fps,
        crop_size=crop_size,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        num_mel_bins=num_mel_bins,
        target_length=target_length,
        audio_sample_rate=sample_rate,
        audio_mean=audio_mean,
        audio_std=audio_std,
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
                use_audio=use_audio,
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
            ema_models=None,
            use_audio=use_audio,
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
    use_audio=False,
):
    """Run one epoch of training or validation for video / audio-video classification."""
    if aug_config is None:
        aug_config = {}

    use_mixup = aug_config.get("use_mixup", False) and training
    mixup_alpha = aug_config.get("mixup_alpha", 0.4)

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
            labels = labels.to(device)

            # Extract video data: av_data["video"] is (buffer, clip_indices)
            # buffer is a list of clips, each clip is [B, C, T, H, W]
            video_tuple = av_data["video"]
            buffer = video_tuple[0]  # list of clips
            video_data = buffer[0].to(device, non_blocking=True)  # [B, C, T, H, W]

            if itr == 0:
                logger.info(f"video_data shape: {video_data.shape}")

            # Extract audio data if needed
            if use_audio:
                audio_data = av_data["audio"].to(device, non_blocking=True)
                if itr == 0:
                    logger.info(f"audio_data shape: {audio_data.shape}")

            # Apply mixup on labels (video augmentation is handled by transforms)
            if use_mixup:
                # For audio-video, we could mix both, but mixup on labels is sufficient
                # since the encoder is frozen and we only train the classifier
                _, labels_for_loss, _ = mixup_data(
                    video_data.view(video_data.size(0), -1),  # dummy for shuffle index
                    labels,
                    alpha=mixup_alpha,
                )
                # NOTE: We only mix labels, not the actual encoder inputs,
                # because the encoder is frozen. Mixing high-dimensional video
                # in pixel space is not standard practice.
                labels_for_loss = labels_for_loss
            else:
                labels_for_loss = labels

            # Forward through encoder
            with torch.no_grad():
                if use_audio:
                    outputs = encoder(audio_x=audio_data, video_x=video_data)
                else:
                    outputs = encoder(video_x=video_data)

            # Forward through classifiers
            if training:
                classifier_outputs = [c(outputs) for c in classifiers]
            else:
                with torch.no_grad():
                    classifier_outputs = [c(outputs) for c in classifiers]

        # Compute losses
        losses = [criterion(co, labels_for_loss) for co in classifier_outputs]

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
    csv_path,
    frames_per_clip=16,
    fps=4,
    crop_size=224,
    batch_size=48,
    world_size=1,
    rank=0,
    training=False,
    num_workers=12,
    # Audio params
    num_mel_bins=128,
    target_length=1024,
    audio_sample_rate=16000,
    audio_mean=-4.268,
    audio_std=4.569,
):
    """
    Create dataloader for AudioSet with audio+video data.

    Uses AudioSetEval dataset which returns both audio spectrograms and video frames.
    Video transforms are hardcoded: random resized crop + horizontal flip for training,
    center crop (no flip, no random resize) for validation.

    Args:
        csv_path: Path to CSV with audio_path, video_path, labels
        frames_per_clip: Number of video frames per clip
        fps: Frames per second for video sampling
        crop_size: Spatial crop size for video frames
        batch_size: Batch size per GPU
        world_size: Number of distributed processes
        rank: Current process rank
        training: Whether in training mode
        num_workers: Number of dataloader workers

    Returns:
        (data_loader, sampler) tuple
    """
    if csv_path is None:
        raise ValueError("csv_path is required. " "Check your config file for train_csv/val_csv.")

    logger.info(f"Creating {'train' if training else 'val'} dataloader")
    logger.info(f"  csv_path: {csv_path}")

    # Video transforms
    if training:
        video_transform = make_transforms(
            random_horizontal_flip=True,
            random_resize_aspect_ratio=(3 / 4, 4 / 3),
            random_resize_scale=(0.8, 1.0),
            crop_size=crop_size,
        )
    else:
        video_transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            crop_size=crop_size,
        )

    dataset = AudioSetEval(
        data_paths=csv_path,
        transform=video_transform,
        frame_step=None,
        fps=fps,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=None,
        # Audio params
        num_mel_bins=num_mel_bins,
        target_length=target_length,
        audio_sample_rate=audio_sample_rate,
        audio_mean=audio_mean,
        audio_std=audio_std,
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

    logger.info(f"  Dataset size: {len(dataset)} samples")

    return data_loader, sampler


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

# Video-only eval (single GPU):
"""
python -m evals.audio_classification_frozen.eval_video \
    --config configs/eval/audioset20k/as20k-video.yaml
"""

# Audio-video eval (single GPU):
"""
python -m evals.audio_classification_frozen.eval_video \
    --config configs/eval/audioset20k/as20k-audio-video.yaml
"""

# Distributed (via main_distributed), substituting your own SLURM account/qos:
"""
python -m evals.main_distributed \
    --fname configs/eval/audioset20k/as20k-video.yaml \
    --time 6000 \
    --account <your_slurm_account> --qos <your_slurm_qos>
"""

"""
python -m evals.main_distributed \
    --fname configs/eval/audioset20k/as20k-audio-video.yaml \
    --time 6000 \
    --account <your_slurm_account> --qos <your_slurm_qos>
"""


if __name__ == "__main__":
    """
    Direct execution for single-GPU testing.

    Usage:
        python -m evals.audio_classification_frozen.eval_video \
            --config configs/eval/audioset/audioset20k_video_only.yaml

        python -m evals.audio_classification_frozen.eval_video \
            --config configs/eval/audioset/audioset20k_audio_video.yaml
    """
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Video / Audio-Video Classification Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--val_only", action="store_true", help="Only run validation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override num epochs")
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.val_only:
        config["val_only"] = True
    if args.checkpoint:
        config["model_kwargs"]["checkpoint"] = args.checkpoint
    if args.batch_size:
        config["experiment"]["optimization"]["batch_size"] = args.batch_size
    if args.num_epochs:
        config["experiment"]["optimization"]["num_epochs"] = args.num_epochs

    main(args_eval=config, resume_preempt=False)
