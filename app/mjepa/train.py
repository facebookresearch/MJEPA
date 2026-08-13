# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time
from contextlib import ExitStack, nullcontext

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.mjepa.transforms import make_transforms
from app.mjepa.utils import init_audio_video_model, init_opt, load_checkpoint, normalize_and_concat
from src.datasets.data_manager import init_data, init_video_only_data
from src.masks.audio_video_random_multiseqmultiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.masks.video_only_multiseqmultiblock3d import VideoOnlyMaskCollator
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

# --
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
# --

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    # hier stuff
    levels_predictor = cfgs_model.get("levels_predictor", 4)  # By default, we have a multi-level predictor!!

    if model_name == "vit_large":
        embed_dim_encoder = 1024
    elif model_name == "vit_huge":
        embed_dim_encoder = 1280
    elif model_name == "vit_giant":
        embed_dim_encoder = 1408
    else:
        assert False, f"Model name not recognized: {model_name}"

    # Audio model
    audio_patch_size = cfgs_model.get("audio_patch_size")
    audio_img_size = cfgs_model.get("audio_input_size")
    audio_in_chans = cfgs_model.get("audio_in_chans")

    # a2v2
    audio_feature_pooling = cfgs_model.get("audio_feature_pooling")
    video_feature_pooling = cfgs_model.get("video_feature_pooling")

    cross_modal_pooling = cfgs_model.get("cross_modal_pooling")

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    # audio params
    num_mel_bins = cfgs_data.get("num_mel_bins")
    target_length = cfgs_data.get("target_length")
    audio_sample_rate = cfgs_data.get("audio_sample_rate")
    audio_mean = cfgs_data.get("audio_mean")
    audio_std = cfgs_data.get("audio_std")
    # dataloader params
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- VIDEO-ONLY DATA (K710, SSv2, etc.)
    cfgs_data_video_only = args.get("data_video_only", None)
    use_video_only_data = cfgs_data_video_only is not None
    # GPU split ratio for video-only processing (fraction of GPUs for video-only)
    video_only_rank_ratio = 0.0
    if use_video_only_data:
        video_only_dataset_paths = cfgs_data_video_only.get("datasets", [])
        video_only_datasets_weights = cfgs_data_video_only.get("datasets_weights")
        video_only_dataset_fpcs = cfgs_data_video_only.get("dataset_fpcs")
        video_only_batch_size = cfgs_data_video_only.get("batch_size", batch_size)
        video_only_fps = cfgs_data_video_only.get("fps", fps)
        video_only_crop_size = cfgs_data_video_only.get("crop_size", crop_size)
        video_only_patch_size = cfgs_data_video_only.get("patch_size", patch_size)
        video_only_tubelet_size = cfgs_data_video_only.get("tubelet_size", tubelet_size)
        video_only_pin_mem = cfgs_data_video_only.get("pin_mem", pin_mem)
        video_only_num_workers = cfgs_data_video_only.get("num_workers", num_workers)
        video_only_persistent_workers = cfgs_data_video_only.get("persistent_workers", persistent_workers)
        # GPU split ratio: fraction of total GPUs dedicated to video-only processing
        # e.g., 0.5 means 50% of GPUs process video-only, 50% process audio-video
        video_only_rank_ratio = cfgs_data_video_only.get("rank_ratio", 0.0)
        logger.info(f"Video-only datasets enabled: {video_only_dataset_paths}")
        logger.info(f"Video-only weights: {video_only_datasets_weights}")
        logger.info(f"Video-only GPU rank ratio: {video_only_rank_ratio}")

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    predict_all = cfgs_loss.get("predict_all", False)
    # Loss coefficients for ablation studies
    coef_v2v = cfgs_loss.get("coef_v2v", 1.0)
    coef_a2a = cfgs_loss.get("coef_a2a", 1.0)
    coef_av2av = cfgs_loss.get("coef_av2av", 1.0)
    coef_a2v = cfgs_loss.get("coef_a2v", 1.0)
    coef_v2a = cfgs_loss.get("coef_v2a", 1.0)
    coef_av2a = cfgs_loss.get("coef_av2a", 1.0)
    coef_av2v = cfgs_loss.get("coef_av2v", 1.0)
    coef_a2av = cfgs_loss.get("coef_a2av", 1.0)
    coef_v2av = cfgs_loss.get("coef_v2av", 1.0)
    # Flag to skip heavy AV forward passes if all AV losses are disabled
    skip_av_forward = (
        coef_av2av == 0.0 and coef_av2a == 0.0 and coef_av2v == 0.0 and coef_a2av == 0.0 and coef_v2av == 0.0
    )
    if skip_av_forward:
        logger.info("All AV loss coefficients are 0. Skipping AV forward passes.")

    # Flag to skip video forward passes (for audio-only ablation)
    # Video forward needed for: v2v_loss, a2v_loss, v2a_loss, v2av_loss, av2v_loss
    skip_video_forward = (
        coef_v2v == 0.0
        and coef_a2v == 0.0
        and coef_v2a == 0.0
        and coef_v2av == 0.0
        and coef_av2v == 0.0
        and coef_av2av == 0.0
    )
    if skip_video_forward:
        logger.info("All video-related loss coefficients are 0. Skipping video forward passes.")

    # Flag to skip audio forward passes (for video-only ablation)
    # Audio forward needed for: a2a_loss, a2v_loss, v2a_loss, a2av_loss, av2a_loss
    skip_audio_forward = (
        coef_a2a == 0.0
        and coef_a2v == 0.0
        and coef_v2a == 0.0
        and coef_a2av == 0.0
        and coef_av2a == 0.0
        and coef_av2av == 0.0
    )
    if skip_audio_forward:
        logger.info("All audio-related loss coefficients are 0. Skipping audio forward passes.")

    # Video-only loss coefficient
    coef_video_only_v2v = cfgs_loss.get("coef_video_only_v2v", 1.0)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    accumulation_steps = cfgs_opt.get("accumulation_steps", 1)
    # -- ANNEALING/COOLDOWN
    is_anneal = cfgs_opt.get("is_anneal", False)
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)
    if is_anneal and anneal_ckpt is None:
        raise ValueError("Must specify anneal_ckpt if is_anneal is True")
    resume_anneal = cfgs_opt.get("resume_anneal", False) or (is_anneal and resume_preempt)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # ----------------------------------------------------------------------- #
    # GPU SPLIT SETUP (following gold standard pattern from vjepa_batched_hier)
    # ----------------------------------------------------------------------- #
    # Key insight: All GPUs run the SAME train_step() with IDENTICAL computation
    # graphs. The only difference is the DATA they process. This allows DDP to
    # work automatically without manual gradient synchronization.
    #
    # - Audio-video GPUs (lower ranks): Process AudioSet with real audio+video
    # - Video-only GPUs (higher ranks): Process K710/SSv2/HT100M with video +
    #   DUMMY audio (zeros) to maintain identical computation graph
    # ----------------------------------------------------------------------- #
    is_audio_video_rank = True  # Default: all GPUs process audio-video
    is_video_only_rank = False
    audio_video_world_size = world_size
    video_only_world_size = 0
    data_rank = rank
    data_world_size = world_size

    # Per-rank loss coefficients (will be modified for video-only ranks)
    local_coef_v2v = coef_v2v
    local_coef_a2a = coef_a2a
    local_coef_av2av = coef_av2av
    local_coef_a2v = coef_a2v
    local_coef_v2a = coef_v2a
    local_coef_av2a = coef_av2a
    local_coef_av2v = coef_av2v
    local_coef_a2av = coef_a2av
    local_coef_v2av = coef_v2av

    if use_video_only_data and video_only_rank_ratio > 0:
        # Calculate number of GPUs for each modality
        video_only_world_size = int(world_size * video_only_rank_ratio)
        audio_video_world_size = world_size - video_only_world_size

        if video_only_world_size == 0:
            logger.warning(
                f"video_only_rank_ratio={video_only_rank_ratio} results in 0 "
                "video-only GPUs. All GPUs will process audio-video."
            )
        elif audio_video_world_size == 0:
            raise ValueError(
                f"video_only_rank_ratio={video_only_rank_ratio} results in 0 "
                "audio-video GPUs. This is not supported."
            )

        # Use config batch sizes directly (no rescaling)
        # batch_size = per-GPU batch size for audio-video
        # video_only_batch_size = per-GPU batch size for video-only
        logger.info(
            f"Using config batch sizes directly: " f"AV batch_size={batch_size}, VO batch_size={video_only_batch_size}"
        )

        # Determine which group this GPU belongs to (following gold standard pattern)
        # Lower ranks: audio-video, Higher ranks: video-only
        if rank < audio_video_world_size:
            # This GPU processes AudioSet (audio-video)
            is_audio_video_rank = True
            is_video_only_rank = False
            data_rank = rank
            data_world_size = audio_video_world_size
            # Use standard audio-video loss coefficients
            local_coef_v2v = coef_v2v
            local_coef_a2a = coef_a2a
            local_coef_av2av = coef_av2av
            local_coef_a2v = coef_a2v
            local_coef_v2a = coef_v2a
            local_coef_av2a = coef_av2a
            local_coef_av2v = coef_av2v
            local_coef_a2av = coef_a2av
            local_coef_v2av = coef_v2av
        else:
            # This GPU processes video-only (K710, SSv2, etc.)
            # Zero out all audio-related losses, only keep v2v loss
            is_audio_video_rank = False
            is_video_only_rank = True
            data_rank = rank - audio_video_world_size
            data_world_size = video_only_world_size
            # For video-only: only v2v loss matters, zero out audio losses
            local_coef_v2v = coef_video_only_v2v
            local_coef_a2a = 0.0
            local_coef_av2av = 0.0
            local_coef_a2v = 0.0
            local_coef_v2a = 0.0
            local_coef_av2a = 0.0
            local_coef_av2v = 0.0
            local_coef_a2av = 0.0
            local_coef_v2av = 0.0
            # NOTE: Video-only GPUs still run full audio-video forward pass with
            # dummy audio to maintain identical computation graph for DDP sync.
            # This uses more memory than the old manual sync approach.
            # If OOM, reduce video_only_batch_size in config.

        logger.info(
            f"GPU Split Mode (Gold Standard Pattern):\n"
            f"  - Total GPUs: {world_size}\n"
            f"  - Audio-video GPUs: {audio_video_world_size} (ranks 0-{audio_video_world_size-1})\n"
            f"  - Video-only GPUs: {video_only_world_size} (ranks {audio_video_world_size}-{world_size-1})\n"
            f"  - This rank ({rank}): {'audio-video' if is_audio_video_rank else 'video-only'}\n"
            f"  - Data rank: {data_rank}, Data world size: {data_world_size}\n"
            f"  - Audio-video batch size: {batch_size if is_audio_video_rank else 'N/A'}\n"
            f"  - Video-only batch size: {video_only_batch_size if is_video_only_rank else 'N/A'}\n"
            f"  - Local loss coefficients: v2v={local_coef_v2v}, a2a={local_coef_a2a}, "
            f"av2av={local_coef_av2av}, a2v={local_coef_a2v}, v2a={local_coef_v2a}"
        )

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_file = "latest.pt"
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        if is_anneal:
            # For annealing: load from latest if resuming, otherwise load from anneal_ckpt
            if os.path.exists(latest_path) and resume_anneal:
                load_path = latest_path
            else:
                load_path = anneal_ckpt
                resume_anneal = False  # Mark that we're starting fresh anneal
        else:
            # Normal pre-training: load from r_file or latest
            load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    # -- init model
    (
        encoder,
        predictor,
        a2v_predictor,
        a2av_predictor,
        v2a_predictor,
        v2av_predictor,
        av2a_predictor,
        av2v_predictor,
    ) = init_audio_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(cfgs_mask) * len(dataset_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        video_patch_size=patch_size,
        video_max_num_frames=max_num_frames,
        video_tubelet_size=tubelet_size,
        model_name=model_name,
        video_img_size=(crop_size, crop_size),
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        # audio params,
        audio_patch_size=audio_patch_size,
        audio_img_size=audio_img_size,
        audio_in_chans=audio_in_chans,
        audio_feature_pooling=audio_feature_pooling,
        video_feature_pooling=video_feature_pooling,
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictors.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()
        a2v_predictor.compile()
        a2av_predictor.compile()
        v2a_predictor.compile()
        v2av_predictor.compile()
        av2a_predictor.compile()
        av2v_predictor.compile()

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    # Following gold standard pattern: each GPU initializes ONE dataloader
    # based on its rank assignment. The key difference is that video-only GPUs
    # will create dummy audio data to maintain identical computation graph.
    if is_video_only_rank:
        # Video-only ranks: use VideoOnlyMaskCollator
        video_only_mask_collator = VideoOnlyMaskCollator(
            cfgs_mask=cfgs_mask,
            dataset_fpcs=video_only_dataset_fpcs,
            crop_size=video_only_crop_size,
            patch_size=video_only_patch_size,
            tubelet_size=video_only_tubelet_size,
        )
        video_only_transform = make_transforms(
            random_horizontal_flip=True,
            random_resize_aspect_ratio=ar_range,
            random_resize_scale=rr_scale,
            reprob=reprob,
            auto_augment=use_aa,
            motion_shift=motion_shift,
            crop_size=video_only_crop_size,
        )
        unsupervised_loader, unsupervised_sampler = init_video_only_data(
            root_path=video_only_dataset_paths,
            batch_size=video_only_batch_size,
            training=True,
            dataset_fpcs=video_only_dataset_fpcs,
            fps=video_only_fps,
            transform=video_only_transform,
            rank=data_rank,
            world_size=data_world_size,
            datasets_weights=video_only_datasets_weights,
            persistent_workers=video_only_persistent_workers,
            collator=video_only_mask_collator,
            num_workers=video_only_num_workers,
            pin_mem=video_only_pin_mem,
            log_dir=None,
        )
        current_batch_size = video_only_batch_size
        logger.info(f"Rank {rank}: Initialized video-only dataloader (data_rank={data_rank})")
    else:
        # Audio-video ranks: use full MaskCollator
        unsupervised_loader, unsupervised_sampler = init_data(
            data=dataset_type,
            root_path=dataset_paths,
            batch_size=batch_size,
            training=True,
            dataset_fpcs=dataset_fpcs,
            fps=fps,
            transform=transform,
            rank=data_rank,
            world_size=data_world_size,
            datasets_weights=datasets_weights,
            persistent_workers=persistent_workers,
            collator=mask_collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            log_dir=None,
            # audio params,
            num_mel_bins=num_mel_bins,
            target_length=target_length,
            audio_sample_rate=audio_sample_rate,
            audio_mean=audio_mean,
            audio_std=audio_std,
        )
        current_batch_size = batch_size
        logger.info(f"Rank {rank}: Initialized audio-video dataloader (data_rank={data_rank})")

    # Compute dataset length
    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        _dlen = unsupervised_loader.num_batches

    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")
    logger.info(f"gradient accumulation steps: {accumulation_steps}")
    logger.info(f"effective batch size: {batch_size * world_size * accumulation_steps}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        a2v_predictor=a2v_predictor,
        a2av_predictor=a2av_predictor,
        v2a_predictor=v2a_predictor,
        v2av_predictor=v2av_predictor,
        av2a_predictor=av2a_predictor,
        av2v_predictor=av2v_predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
        is_anneal=is_anneal,
    )
    # With the unified train_step approach (gold standard pattern), all GPUs run
    # identical forward passes, so all parameters are used. We only need
    # find_unused_parameters when skipping specific forward passes for ablation
    # OR when using gradient accumulation (which can cause some params to appear unused
    # between accumulation steps due to the way DDP tracks gradient completion).
    find_unused = skip_video_forward or skip_audio_forward or accumulation_steps > 1
    if find_unused:
        if skip_video_forward or skip_audio_forward:
            logger.info("Enabling find_unused_parameters=True for DDP (ablation mode)")
        elif accumulation_steps > 1:
            logger.info("Enabling find_unused_parameters=True for DDP (gradient accumulation)")

    # Use static_graph=True for better performance since all GPUs run identical graphs
    # EXCEPT when using gradient accumulation, which is incompatible with static_graph
    use_static_graph = not find_unused and accumulation_steps == 1
    if accumulation_steps > 1:
        logger.info("Disabling static_graph for DDP due to gradient accumulation")
    encoder = DistributedDataParallel(encoder, static_graph=use_static_graph, find_unused_parameters=find_unused)
    predictor = DistributedDataParallel(predictor, static_graph=use_static_graph, find_unused_parameters=find_unused)
    target_encoder = DistributedDataParallel(target_encoder, static_graph=use_static_graph)

    # Wrap cross-modal predictors with DDP
    a2v_predictor = DistributedDataParallel(
        a2v_predictor, static_graph=use_static_graph, find_unused_parameters=find_unused
    )
    a2av_predictor = DistributedDataParallel(
        a2av_predictor, static_graph=use_static_graph, find_unused_parameters=find_unused
    )
    v2a_predictor = DistributedDataParallel(
        v2a_predictor, static_graph=use_static_graph, find_unused_parameters=find_unused
    )
    v2av_predictor = DistributedDataParallel(
        v2av_predictor, static_graph=use_static_graph, find_unused_parameters=find_unused
    )
    av2a_predictor = DistributedDataParallel(
        av2a_predictor, static_graph=use_static_graph, find_unused_parameters=find_unused
    )
    av2v_predictor = DistributedDataParallel(
        av2v_predictor, static_graph=use_static_graph, find_unused_parameters=find_unused
    )

    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    # -- load training checkpoint
    if load_model or os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            a2v_predictor,
            a2av_predictor,
            v2a_predictor,
            v2av_predictor,
            av2a_predictor,
            av2v_predictor,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            a2v_predictor=a2v_predictor,
            a2av_predictor=a2av_predictor,
            v2a_predictor=v2a_predictor,
            v2av_predictor=v2av_predictor,
            av2a_predictor=av2a_predictor,
            av2v_predictor=av2v_predictor,
            opt=optimizer,
            scaler=scaler,
            is_anneal=is_anneal and not resume_anneal,
        )
        # Only fast-forward schedulers if NOT starting fresh anneal
        if not is_anneal or resume_anneal:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "a2v_predictor": a2v_predictor.state_dict(),
            "a2av_predictor": a2av_predictor.state_dict(),
            "v2a_predictor": v2a_predictor.state_dict(),
            "v2av_predictor": v2av_predictor.state_dict(),
            "av2a_predictor": av2a_predictor.state_dict(),
            "av2v_predictor": av2v_predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    # Following gold standard: each GPU has ONE loader initialized earlier
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        # -- update distributed-data-loader epoch

        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        effective_loss_meter = AverageMeter()
        effective_av_loss_meter = AverageMeter()
        effective_vo_loss_meter = AverageMeter()
        audio_video_loss_meter = AverageMeter()
        video_only_loss_meter = AverageMeter()
        v2v_loss_meter = AverageMeter()
        a2a_loss_meter = AverageMeter()
        av2av_loss_meter = AverageMeter()
        a2v_loss_meter = AverageMeter()
        a2av_loss_meter = AverageMeter()
        v2a_loss_meter = AverageMeter()
        v2av_loss_meter = AverageMeter()
        av2a_loss_meter = AverageMeter()
        av2v_loss_meter = AverageMeter()
        mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            # Load data - following gold standard pattern where each GPU has ONE loader
            iter_retries = 0
            iter_successful = False
            sample = None

            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loader. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(
                            f"Encountered exception when loading data " f"(num retries {iter_retries}):\n{e}"
                        )
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        raise RuntimeError(f"Exceeded max retries ({NUM_RETRIES}) when loading data.") from e

            # -------------------------------------------------------------------
            # UNIFIED DATA LOADING (Gold Standard Pattern)
            # -------------------------------------------------------------------
            # All GPUs process their data into a unified format with the SAME
            # structure. Video-only GPUs create DUMMY audio tensors to maintain
            # identical computation graphs, allowing DDP to sync automatically.
            # -------------------------------------------------------------------

            def load_clips_unified():
                """
                Load clips into a unified format for ALL GPUs.
                Video-only GPUs create dummy audio tensors to maintain identical
                computation graph structure with audio-video GPUs.
                """
                all_clips = []
                all_masks_enc = []
                all_masks_pred = []
                all_audio = []
                all_audio_masks_enc = []
                all_audio_masks_pred = []

                if is_audio_video_rank:
                    # Audio-video data: unpack all components
                    for fpc_sample in sample:
                        udata, masks_enc, masks_pred, audio_masks_enc_sample, audio_masks_pred_sample = fpc_sample
                        video_tensor = udata["video"][0][0].to(device, non_blocking=True)
                        all_clips.append(video_tensor)
                        all_audio.append(udata["audio"].to(device, non_blocking=True))
                        all_masks_enc.append([m.to(device, non_blocking=True) for m in masks_enc])
                        all_masks_pred.append([m.to(device, non_blocking=True) for m in masks_pred])
                        all_audio_masks_enc.append([m.to(device, non_blocking=True) for m in audio_masks_enc_sample])
                        all_audio_masks_pred.append([m.to(device, non_blocking=True) for m in audio_masks_pred_sample])
                else:
                    # Video-only data: create DUMMY audio tensors for identical graph
                    for fpc_sample in sample:
                        udata, masks_enc, masks_pred = fpc_sample
                        video_tensor = udata["video"][0][0].to(device, non_blocking=True)
                        all_clips.append(video_tensor)
                        all_masks_enc.append([m.to(device, non_blocking=True) for m in masks_enc])
                        all_masks_pred.append([m.to(device, non_blocking=True) for m in masks_pred])

                        # Create dummy audio with correct shape
                        # Audio shape: [batch_size, channels, height, width]
                        # From config: audio_input_size=[1024, 128], audio_in_chans=1
                        bs = video_tensor.shape[0]
                        dummy_audio = torch.zeros(
                            bs,
                            audio_in_chans,
                            audio_img_size[0],
                            audio_img_size[1],
                            device=device,
                            dtype=video_tensor.dtype,
                        )
                        all_audio.append(dummy_audio)

                        # Create dummy audio masks (select minimal patches)
                        # These masks have same structure as real masks but select minimal patches
                        num_masks_enc = len(masks_enc)
                        num_masks_pred = len(masks_pred)
                        dummy_audio_mask_enc = [
                            torch.zeros(bs, 1, dtype=torch.long, device=device) for _ in range(num_masks_enc)
                        ]
                        dummy_audio_mask_pred = [
                            torch.zeros(bs, 1, dtype=torch.long, device=device) for _ in range(num_masks_pred)
                        ]
                        all_audio_masks_enc.append(dummy_audio_mask_enc)
                        all_audio_masks_pred.append(dummy_audio_mask_pred)

                return (all_clips, all_audio, all_masks_enc, all_masks_pred, all_audio_masks_enc, all_audio_masks_pred)

            # Update mask meters
            for fpc_idx, fpc_sample in enumerate(sample):
                if is_audio_video_rank:
                    video_data = fpc_sample[0]["video"]
                else:
                    video_data = fpc_sample[0]["video"]
                bs = video_data[-1][0].size()[0]
                fpc = video_data[-1][0].size()[1]
                if fpc in mask_meters:
                    mask_meters[fpc].update(bs / current_batch_size)

            clips, audio_spectrograms, video_masks_enc, video_masks_pred, audio_masks_enc, audio_masks_pred = (
                load_clips_unified()
            )

            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def audio_video_train_step():
                # --

                # video first
                def forward_video_target(c):
                    with torch.no_grad():
                        _, vh = target_encoder(video_x=c, training_mode=True)
                        if levels_predictor > 1:
                            vh_new = []
                            for hi in vh:
                                hi_0 = F.layer_norm(hi[:, :, :embed_dim_encoder], (embed_dim_encoder,))
                                hi_1 = F.layer_norm(
                                    hi[:, :, embed_dim_encoder : 2 * embed_dim_encoder], (embed_dim_encoder,)
                                )
                                hi_2 = F.layer_norm(
                                    hi[:, :, 2 * embed_dim_encoder : 3 * embed_dim_encoder], (embed_dim_encoder,)
                                )
                                hi_3 = F.layer_norm(hi[:, :, -embed_dim_encoder:], (embed_dim_encoder,))
                                hi_norm = torch.cat([hi_0, hi_1, hi_2, hi_3], dim=-1)
                                vh_new.append(hi_norm)
                        else:
                            vh_new = [F.layer_norm(hi, (hi.size(-1),)) for hi in vh]
                        return vh_new

                def forward_video_context(c, video_masks_enc, video_masks_pred):
                    # MultiSeqWrapper makes it a list
                    _, vz = encoder(video_x=c, video_masks=video_masks_enc, training_mode=True)
                    # MultiSeqWrapperPred makes it a list
                    _, v2v_z_pred, _, v2v_z_context = predictor(
                        video_x=vz, video_masks_x=video_masks_enc, video_masks_y=video_masks_pred
                    )

                    # c is of length num of fscs = num of datasets in this case = 1
                    # vz is seq of length 1 = len (c)
                    # each element in vz is a seq of len(type of masks) = 2 in our current config
                    # for each type of mask: it is of shape [bs, tokens, emb_dim]

                    # v2v_z is same as vz but for pred tokens

                    # normalize_predictor nested
                    v2v_z_pred = [[normalize_and_concat(z, embed_dim_encoder) for z in inner] for inner in v2v_z_pred]

                    if predict_all:
                        assert v2v_z_context is not None
                        v2v_z_context = [
                            [normalize_and_concat(z, embed_dim_encoder) for z in inner] for inner in v2v_z_context
                        ]

                    # using last layer features to predict audio and av last layer features
                    # vz as input which is num_fscs x num_of_mask_types x batch data
                    v2a_z = [[] for _ in vz]
                    for i, vzi in enumerate(vz):
                        for vzij in vzi:
                            # 4 layers features are concatenated at emb dim
                            last_layer_features = vzij[:, :, -embed_dim_encoder:]
                            v2a_z[i] += [v2a_predictor(last_layer_features)]

                    v2av_z = [[] for _ in vz]
                    for i, vzi in enumerate(vz):
                        for vzij in vzi:
                            # 4 layers features are concatenated at emb dim
                            last_layer_features = vzij[:, :, -embed_dim_encoder:]
                            v2av_z[i] += [v2av_predictor(last_layer_features)]

                    return v2v_z_pred, v2v_z_context, v2a_z, v2av_z

                # audio now
                def forward_audio_target(audio_spectrograms):
                    # this is a list of batches, one for each dataset
                    with torch.no_grad():
                        ah, _ = target_encoder(audio_x=audio_spectrograms, training_mode=True)
                        if levels_predictor > 1:
                            ah_new = []
                            for hi in ah:
                                hi_0 = F.layer_norm(hi[:, :, :embed_dim_encoder], (embed_dim_encoder,))
                                hi_1 = F.layer_norm(
                                    hi[:, :, embed_dim_encoder : 2 * embed_dim_encoder], (embed_dim_encoder,)
                                )
                                hi_2 = F.layer_norm(
                                    hi[:, :, 2 * embed_dim_encoder : 3 * embed_dim_encoder], (embed_dim_encoder,)
                                )
                                hi_3 = F.layer_norm(hi[:, :, -embed_dim_encoder:], (embed_dim_encoder,))
                                hi_norm = torch.cat([hi_0, hi_1, hi_2, hi_3], dim=-1)
                                ah_new.append(hi_norm)
                        else:
                            ah_new = [F.layer_norm(hi, (hi.size(-1),)) for hi in ah]
                        return ah_new

                def forward_audio_context(audio_spectrograms, audio_masks_enc, audio_masks_pred):
                    az, _ = encoder(audio_x=audio_spectrograms, audio_masks=audio_masks_enc, training_mode=True)
                    a2a_z_pred, _, a2a_z_context, _ = predictor(
                        audio_x=az, audio_masks_x=audio_masks_enc, audio_masks_y=audio_masks_pred
                    )

                    # normalize_predictor nested
                    a2a_z_pred = [[normalize_and_concat(z, embed_dim_encoder) for z in inner] for inner in a2a_z_pred]

                    if predict_all:
                        assert a2a_z_context is not None
                        a2a_z_context = [
                            [normalize_and_concat(z, embed_dim_encoder) for z in inner] for inner in a2a_z_context
                        ]

                    # using last layer features to predict video and av last layer features
                    # az as input which is num_fscs x num_of_mask_types x batch data
                    a2v_z = [[] for _ in az]
                    for i, azi in enumerate(az):  # over datasets
                        for azij in azi:  # over mask types
                            last_layer_features = azij[:, :, -embed_dim_encoder:]
                            a2v_z[i] += [a2v_predictor(last_layer_features)]

                    a2av_z = [[] for _ in az]
                    for i, azi in enumerate(az):  # over datasets
                        for azij in azi:  # over mask types
                            last_layer_features = azij[:, :, -embed_dim_encoder:]
                            a2av_z[i] += [a2av_predictor(last_layer_features)]

                    return a2a_z_pred, a2a_z_context, a2v_z, a2av_z

                # now audio video combined
                def forward_audio_video_target(audio_spectrograms, c):
                    with torch.no_grad():
                        av_ah, av_vh = target_encoder(
                            audio_x=audio_spectrograms,
                            video_x=c,
                            training_mode=True,
                        )
                        if levels_predictor > 1:
                            av_ah_new = []
                            for hi in av_ah:
                                hi_0 = F.layer_norm(hi[:, :, :embed_dim_encoder], (embed_dim_encoder,))
                                hi_1 = F.layer_norm(
                                    hi[:, :, embed_dim_encoder : 2 * embed_dim_encoder], (embed_dim_encoder,)
                                )
                                hi_2 = F.layer_norm(
                                    hi[:, :, 2 * embed_dim_encoder : 3 * embed_dim_encoder], (embed_dim_encoder,)
                                )
                                hi_3 = F.layer_norm(hi[:, :, -embed_dim_encoder:], (embed_dim_encoder,))
                                av_ah_new.append(torch.cat([hi_0, hi_1, hi_2, hi_3], dim=-1))
                            av_vh_new = []
                            for hi in av_vh:
                                hi_0 = F.layer_norm(hi[:, :, :embed_dim_encoder], (embed_dim_encoder,))
                                hi_1 = F.layer_norm(
                                    hi[:, :, embed_dim_encoder : 2 * embed_dim_encoder], (embed_dim_encoder,)
                                )
                                hi_2 = F.layer_norm(
                                    hi[:, :, 2 * embed_dim_encoder : 3 * embed_dim_encoder], (embed_dim_encoder,)
                                )
                                hi_3 = F.layer_norm(hi[:, :, -embed_dim_encoder:], (embed_dim_encoder,))
                                av_vh_new.append(torch.cat([hi_0, hi_1, hi_2, hi_3], dim=-1))
                            av_ah, av_vh = av_ah_new, av_vh_new
                        else:
                            av_ah = [F.layer_norm(hi, (hi.size(-1),)) for hi in av_ah]
                            av_vh = [F.layer_norm(hi, (hi.size(-1),)) for hi in av_vh]
                        return av_ah, av_vh

                def forward_audio_video_context(
                    audio_spectrograms,
                    audio_masks_enc,
                    audio_masks_pred,
                    c,
                    video_masks_enc,
                    video_masks_pred,
                ):
                    av_az, av_vz = encoder(
                        audio_x=audio_spectrograms,
                        audio_masks=audio_masks_enc,
                        video_x=c,
                        video_masks=video_masks_enc,
                        training_mode=True,
                    )
                    av_a2a_z_pred, av_v2v_z_pred, av_a2a_z_context, av_v2v_z_context = predictor(
                        audio_x=av_az,
                        audio_masks_x=audio_masks_enc,
                        audio_masks_y=audio_masks_pred,
                        video_x=av_vz,
                        video_masks_x=video_masks_enc,
                        video_masks_y=video_masks_pred,
                    )

                    # normalize_predictor nested
                    av_a2a_z_pred = [
                        [normalize_and_concat(z, embed_dim_encoder) for z in inner] for inner in av_a2a_z_pred
                    ]
                    av_v2v_z_pred = [
                        [normalize_and_concat(z, embed_dim_encoder) for z in inner] for inner in av_v2v_z_pred
                    ]

                    if predict_all:
                        assert av_a2a_z_context is not None
                        assert av_v2v_z_context is not None
                        av_a2a_z_context = [
                            [normalize_and_concat(z, embed_dim_encoder) for z in inner] for inner in av_a2a_z_context
                        ]
                        av_v2v_z_context = [
                            [normalize_and_concat(z, embed_dim_encoder) for z in inner] for inner in av_v2v_z_context
                        ]

                    av2a_z = [[] for _ in av_az]  # just for size
                    av2v_z = [[] for _ in av_vz]  # just for size
                    for i, (azi, vzi) in enumerate(zip(av_az, av_vz)):  # over datasets
                        for azij, vzij in zip(azi, vzi):  # over mask types
                            # Extract last layer features before prediction
                            azi_last = azij[:, :, -embed_dim_encoder:]
                            vzi_last = vzij[:, :, -embed_dim_encoder:]
                            av2a_z[i] += [av2a_predictor(torch.cat([azi_last, vzi_last], dim=1))]
                            av2v_z[i] += [av2v_predictor(torch.cat([azi_last, vzi_last], dim=1))]

                    return av_a2a_z_pred, av_v2v_z_pred, av_a2a_z_context, av_v2v_z_context, av2a_z, av2v_z

                def m2m_loss_fn(z, h, masks_pred):
                    # first for video
                    # Assumption: predictor will have returned only masked tokens for z
                    # Bounds check: mask indices must be within the token grid
                    for hi, mi in zip(h, masks_pred):
                        for mij in mi:
                            max_idx = mij.max().item()
                            num_patches = hi.size(1)
                            if max_idx >= num_patches:
                                raise ValueError(f"mask index {max_idx} >= num_patches {num_patches}")
                    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]

                    loss, n = 0, 0
                    for zi, hi in zip(z, h):
                        for zij, hij in zip(zi, hi):
                            loss += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                            n += 1
                    loss /= n
                    return loss

                def a2v2_loss_fn(z, h):
                    # z is list of list of shape:
                    #   num_datasets x num_of_mask_types x batch data.
                    #   this is pooled batch prediction [bs, emb_dim]
                    # h is list of shape: num_datasets x batch data. h is target features

                    assert len(z) == len(h)

                    if cross_modal_pooling == "mean":
                        loss, n = 0, 0
                        for i in range(len(z)):
                            hi = h[i]  # [bs, target_tokens, emb_dim]
                            hi = torch.mean(hi, dim=1)
                            zi = z[i]  # a list of size num of mask types

                            for j in range(len(zi)):
                                zij = zi[j]  # [bs, emb_dim] predicting pooled features
                                loss += torch.mean(torch.abs(zij - hi[:, -embed_dim_encoder:]) ** loss_exp) / loss_exp
                                n += 1

                        loss /= n
                    else:
                        assert False, "other cross modal pooling not implemented. \
                            Could be attentive shallow encoder on top"

                    return loss

                def a2v2av_loss_fn(z, av_ah, av_vh):
                    assert len(z) == len(av_ah)
                    assert len(z) == len(av_vh)

                    if cross_modal_pooling == "mean":
                        loss, n = 0, 0
                        for i in range(len(z)):
                            ahi = av_ah[i]  # [bs, target_tokens, emb_dim]
                            vhi = av_vh[i]  # [bs, target_tokens, emb_dim]
                            hi = torch.cat([ahi, vhi], dim=1)
                            hi = torch.mean(hi, dim=1)
                            zi = z[i]  # a list of size num of mask types

                            for j in range(len(zi)):
                                zij = zi[j]  # [bs, emb_dim] predicting pooled features
                                loss += torch.mean(torch.abs(zij - hi[:, -embed_dim_encoder:]) ** loss_exp) / loss_exp
                                n += 1

                        loss /= n
                    else:
                        assert False, "other cross modal pooling not implemented. \
                            Could be attentive shallow encoder on top"

                    return loss

                # Step 1. Forward
                with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                    # Conditionally process video (skip for audio-only ablation)
                    if skip_video_forward:
                        vh = None
                        v2v_z, v2a_z, v2av_z = None, None, None
                        v2v_loss = torch.tensor(0.0, device=device)
                    else:
                        vh = forward_video_target(clips)
                        v2v_z, _, v2a_z, v2av_z = forward_video_context(
                            clips, video_masks_enc=video_masks_enc, video_masks_pred=video_masks_pred
                        )
                        v2v_loss = m2m_loss_fn(v2v_z, vh, masks_pred=video_masks_pred)

                    # Conditionally process audio (skip for video-only ablation)
                    if skip_audio_forward:
                        ah = None
                        a2a_z, a2v_z, a2av_z = None, None, None
                        a2a_loss = torch.tensor(0.0, device=device)
                    else:
                        ah = forward_audio_target(audio_spectrograms)
                        a2a_z, _, a2v_z, a2av_z = forward_audio_context(
                            audio_spectrograms, audio_masks_enc=audio_masks_enc, audio_masks_pred=audio_masks_pred
                        )
                        a2a_loss = m2m_loss_fn(a2a_z, ah, masks_pred=audio_masks_pred)

                    # Cross-modal losses (need both modalities)
                    if skip_video_forward or skip_audio_forward:
                        a2v_loss = torch.tensor(0.0, device=device)
                        v2a_loss = torch.tensor(0.0, device=device)
                    else:
                        a2v_loss = a2v2_loss_fn(a2v_z, vh)
                        v2a_loss = a2v2_loss_fn(v2a_z, ah)

                    # AV losses (skip if all AV coefficients are zero or missing modality)
                    if skip_av_forward or skip_video_forward or skip_audio_forward:
                        av2av_loss = torch.tensor(0.0, device=device)
                        av2a_loss = torch.tensor(0.0, device=device)
                        av2v_loss = torch.tensor(0.0, device=device)
                        a2av_loss = torch.tensor(0.0, device=device)
                        v2av_loss = torch.tensor(0.0, device=device)
                    else:
                        av_ah, av_vh = forward_audio_video_target(audio_spectrograms, clips)
                        av_a2a_z, av_v2v_z, _, _, av2a_z, av2v_z = forward_audio_video_context(
                            audio_spectrograms=audio_spectrograms,
                            audio_masks_enc=audio_masks_enc,
                            audio_masks_pred=audio_masks_pred,
                            c=clips,
                            video_masks_enc=video_masks_enc,
                            video_masks_pred=video_masks_pred,
                        )

                        av2av_loss = m2m_loss_fn(av_a2a_z, av_ah, masks_pred=audio_masks_pred) + m2m_loss_fn(
                            av_v2v_z, av_vh, masks_pred=video_masks_pred
                        )
                        av2a_loss = a2v2_loss_fn(av2a_z, ah)
                        av2v_loss = a2v2_loss_fn(av2v_z, vh)

                        a2av_loss = a2v2av_loss_fn(a2av_z, av_ah, av_vh)
                        v2av_loss = a2v2av_loss_fn(v2av_z, av_ah, av_vh)

                    # ------------------Context loss
                    if predict_all:
                        assert False, "not implemented yet"

                    # Use LOCAL coefficients (per-rank) - video-only ranks have audio coeffs = 0
                    loss = (
                        local_coef_v2v * v2v_loss
                        + local_coef_a2a * a2a_loss
                        + local_coef_av2av * av2av_loss
                        + local_coef_a2v * a2v_loss
                        + local_coef_v2a * v2a_loss
                        + local_coef_av2a * av2a_loss
                        + local_coef_av2v * av2v_loss
                        + local_coef_a2av * a2av_loss
                        + local_coef_v2av * v2av_loss
                    )

                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps

                # Step 2. Backward (accumulate gradients, don't step yet)
                if mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                return (
                    float(loss) * accumulation_steps,  # Return unscaled loss for logging
                    float(v2v_loss),
                    float(a2a_loss),
                    float(av2av_loss),
                    float(a2v_loss),
                    float(v2a_loss),
                    float(av2a_loss),
                    float(av2v_loss),
                    float(a2av_loss),
                    float(v2av_loss),
                )

            # -------------------------------------------------------------------
            # UNIFIED TRAINING EXECUTION (Gold Standard Pattern)
            # -------------------------------------------------------------------
            # ALL GPUs run the SAME train_step (audio_video_train_step) with
            # IDENTICAL computation graphs. DDP handles gradient sync automatically.
            #
            # Key differences from manual sync approach:
            # 1. Video-only GPUs use dummy audio (created in load_clips_unified)
            # 2. Per-rank loss coefficients zero out irrelevant losses (local_coef_*)
            # 3. For gradient accumulation, we use no_sync() during accumulation
            #    steps to defer gradient sync until the final step
            # -------------------------------------------------------------------

            # Initialize loss accumulators for gradient accumulation
            accum_train_loss = 0.0
            accum_v2v_loss = 0.0
            accum_a2a_loss = 0.0
            accum_av2av_loss = 0.0
            accum_a2v_loss = 0.0
            accum_v2a_loss = 0.0
            accum_av2a_loss = 0.0
            accum_av2v_loss = 0.0
            accum_a2av_loss = 0.0
            accum_v2av_loss = 0.0
            accum_gpu_etime_ms = 0.0

            # Gradient accumulation loop
            for accum_step in range(accumulation_steps):
                is_last_accum_step = accum_step == accumulation_steps - 1

                # Use no_sync() for non-final accumulation steps to skip gradient all-reduce
                # On the final step, DDP will sync gradients normally
                # We need to apply no_sync() to ALL DDP-wrapped modules
                if accumulation_steps > 1 and not is_last_accum_step:
                    sync_context = ExitStack()
                    sync_context.enter_context(encoder.no_sync())
                    sync_context.enter_context(predictor.no_sync())
                    sync_context.enter_context(a2v_predictor.no_sync())
                    sync_context.enter_context(a2av_predictor.no_sync())
                    sync_context.enter_context(v2a_predictor.no_sync())
                    sync_context.enter_context(v2av_predictor.no_sync())
                    sync_context.enter_context(av2a_predictor.no_sync())
                    sync_context.enter_context(av2v_predictor.no_sync())
                else:
                    sync_context = nullcontext()

                # If accumulation_steps > 1, we need to load new data for each accumulation step
                if accum_step > 0:
                    # Load next batch for accumulation
                    accum_iter_successful = False
                    while not accum_iter_successful:
                        try:
                            sample = next(loader)
                            accum_iter_successful = True
                        except StopIteration:
                            logger.info("Exhausted data loader during accumulation. Refreshing...")
                            unsupervised_sampler.set_epoch(epoch)
                            loader = iter(unsupervised_loader)
                        except Exception as e:
                            logger.warning(f"Exception during accumulation data loading: {e}")
                            time.sleep(5)

                    # Re-run load_clips_unified for the new sample
                    (
                        all_clips,
                        all_masks_enc,
                        all_masks_pred,
                        all_audio,
                        all_audio_masks_enc,
                        all_audio_masks_pred,
                    ), _ = gpu_timer(load_clips_unified)

                # Execute unified train step inside sync context
                # no_sync() prevents gradient all-reduce during intermediate accumulation steps
                with sync_context:
                    (
                        train_loss,
                        v2v_loss,
                        a2a_loss,
                        av2av_loss,
                        a2v_loss,
                        v2a_loss,
                        av2a_loss,
                        av2v_loss,
                        a2av_loss,
                        v2av_loss,
                    ), gpu_etime_ms = gpu_timer(audio_video_train_step)

                # Accumulate losses for logging (average over accumulation steps)
                accum_train_loss += train_loss / accumulation_steps
                accum_v2v_loss += v2v_loss / accumulation_steps
                accum_a2a_loss += a2a_loss / accumulation_steps
                accum_av2av_loss += av2av_loss / accumulation_steps
                accum_a2v_loss += a2v_loss / accumulation_steps
                accum_v2a_loss += v2a_loss / accumulation_steps
                accum_av2a_loss += av2a_loss / accumulation_steps
                accum_av2v_loss += av2v_loss / accumulation_steps
                accum_a2av_loss += a2av_loss / accumulation_steps
                accum_v2av_loss += v2av_loss / accumulation_steps
                accum_gpu_etime_ms += gpu_etime_ms

            # Use accumulated losses for the rest of the iteration
            train_loss = accum_train_loss
            v2v_loss = accum_v2v_loss
            a2a_loss = accum_a2a_loss
            av2av_loss = accum_av2av_loss
            a2v_loss = accum_a2v_loss
            v2a_loss = accum_v2a_loss
            av2a_loss = accum_av2a_loss
            av2v_loss = accum_av2v_loss
            a2av_loss = accum_a2av_loss
            v2av_loss = accum_v2av_loss
            gpu_etime_ms = accum_gpu_etime_ms

            # Step 3. Optimizer step (DDP already synced gradients)
            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()
            if mixed_precision:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            # Step 4. Momentum update of target encoder
            m = next(momentum_scheduler)
            with torch.no_grad():
                params_k = []
                params_q = []
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    params_k.append(param_k)
                    params_q.append(param_q)
                torch._foreach_mul_(params_k, m)
                torch._foreach_add_(params_k, params_q, alpha=1 - m)

            # Set combined losses for logging (video-only loss is 0 for audio-video ranks)
            audio_video_loss = train_loss if is_audio_video_rank else 0.0
            video_only_loss = train_loss if is_video_only_rank else 0.0

            # -------------------------------------------------------------------
            # Lightweight loss sync for logging (only when using GPU split mode)
            # -------------------------------------------------------------------
            # Sync ~10 scalar values across ranks for accurate logging.
            # This is extremely cheap (<1ms) compared to gradient sync.
            # -------------------------------------------------------------------
            av_gpu_time_ms = gpu_etime_ms if is_audio_video_rank else 0.0
            vo_gpu_time_ms = gpu_etime_ms if is_video_only_rank else 0.0

            if use_video_only_data and video_only_rank_ratio > 0:
                # Create tensor with losses and timing to sync
                sync_tensor = torch.tensor(
                    [
                        audio_video_loss,
                        video_only_loss,
                        av_gpu_time_ms,
                        vo_gpu_time_ms,
                    ],
                    device=device,
                    dtype=torch.float32,
                )

                # Sum across all ranks
                torch.distributed.all_reduce(sync_tensor, op=torch.distributed.ReduceOp.SUM)

                # Average by number of contributing ranks
                audio_video_loss = sync_tensor[0].item() / audio_video_world_size
                video_only_loss = sync_tensor[1].item() / video_only_world_size
                av_gpu_time_ms = sync_tensor[2].item() / audio_video_world_size
                vo_gpu_time_ms = sync_tensor[3].item() / video_only_world_size

            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            # Compute effective losses (weighted contributions to gradient)
            # This reflects the true contribution of each loss type after DDP all-reduce
            if use_video_only_data and video_only_rank_ratio > 0:
                effective_av_loss = audio_video_loss * audio_video_world_size / world_size
                effective_vo_loss = video_only_loss * video_only_world_size / world_size
                effective_train_loss = effective_av_loss + effective_vo_loss
            else:
                effective_av_loss = audio_video_loss
                effective_vo_loss = 0.0
                effective_train_loss = train_loss

            loss_meter.update(train_loss)
            effective_loss_meter.update(effective_train_loss)
            effective_av_loss_meter.update(effective_av_loss)
            effective_vo_loss_meter.update(effective_vo_loss)
            audio_video_loss_meter.update(audio_video_loss)
            video_only_loss_meter.update(video_only_loss)
            v2v_loss_meter.update(v2v_loss)
            a2a_loss_meter.update(a2a_loss)
            av2av_loss_meter.update(av2av_loss)
            a2v_loss_meter.update(a2v_loss)
            v2a_loss_meter.update(v2a_loss)
            av2a_loss_meter.update(av2a_loss)
            av2v_loss_meter.update(av2v_loss)
            a2av_loss_meter.update(a2av_loss)
            v2av_loss_meter.update(v2av_loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, train_loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(train_loss) or np.isinf(train_loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f "
                        "masks: %s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            "[" + ", ".join([f"{k}: " + "%.1f" % mask_meters[k].avg for k in mask_meters]) + "]",
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

                    if rank == 0:
                        log_dict = {
                            "train_loss": loss_meter.avg,
                            "effective_train_loss_per_gpu": effective_loss_meter.avg,
                            "audio_video_loss": audio_video_loss_meter.avg,
                            "video_only_loss": video_only_loss_meter.avg,
                            "effective_av_loss": effective_av_loss_meter.avg,
                            "effective_vo_loss": effective_vo_loss_meter.avg,
                            "v2v_loss": v2v_loss_meter.avg,
                            "a2a_loss": a2a_loss_meter.avg,
                            "av2av_loss": av2av_loss_meter.avg,
                            "a2v_loss": a2v_loss_meter.avg,
                            "v2a_loss": v2a_loss_meter.avg,
                            "av2a_loss": av2a_loss_meter.avg,
                            "av2v_loss": av2v_loss_meter.avg,
                            "a2av_loss": a2av_loss_meter.avg,
                            "v2av_loss": v2av_loss_meter.avg,
                            "epoch": epoch,
                            "lr": _new_lr,
                            "gpu_time_ms": gpu_time_meter.avg,
                            "iter_time_ms": iter_time_meter.avg,
                            "data_time_ms": data_elapsed_time_meter.avg,
                        }
                        # Add per-rank-type GPU times for load balancing analysis
                        if use_video_only_data and video_only_rank_ratio > 0:
                            log_dict["av_gpu_time_ms"] = av_gpu_time_ms
                            log_dict["vo_gpu_time_ms"] = vo_gpu_time_ms
                            log_dict["gpu_time_diff_ms"] = av_gpu_time_ms - vo_gpu_time_ms

            log_stats()
            assert not np.isnan(train_loss), "train_loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
