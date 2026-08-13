# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
import warnings

import torch
import torch.nn.functional as F
import yaml

import src.models.mjepa_encoder as mjepa_enc
import src.models.mjepa_predictor as mjepa_pred
from src.audio_models.a2v_predictors import A2V2_MLP_Predictor as a2v2predictor
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, LinearDecaySchedule, WarmupCosineSchedule
from src.utils.tensors import trunc_normal_
from src.utils.wrappers import MJEPAMultiSeqWrapper, MJEPAPredictorHierMultiSeqWrapper

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

MAX_RETRIES = 3


def normalize_and_concat(tensor, embed_dim):
    """Split tensor into 4 chunks of size embed_dim along the last axis,
    apply LayerNorm to each chunk, then concatenate back."""
    chunks = [F.layer_norm(tensor[:, :, i * embed_dim : (i + 1) * embed_dim], (embed_dim,)) for i in range(4)]
    return torch.cat(chunks, dim=2)


def build_eval_args(
    model_name,
    patch_size,
    tubelet_size,
    num_frames,
    logging_folder,
    checkpoint,
    write_tag,
    eval_cfg_paths,
    uniform_power=False,
    use_sdpa=False,
    clip_duration=None,
    use_silu=False,
    wide_silu=True,
    tag=None,
):
    """
    Helper function to parse the pre-training configs to construct the
    evaluation configs, return as a list of eval configs.
    """
    # By convention, the pre-training config should specify any required evals
    # in the 'evals' key
    if eval_cfg_paths is None:
        logger.info("No evaluations specified!")
        return

    eval_nodes = None
    eval_tasks_per_node = None
    args_eval = []
    for i, f in enumerate(eval_cfg_paths):
        with open(f, "r") as y_file:
            _args = yaml.load(y_file, Loader=yaml.FullLoader)
            _tag = _args.get("tag", "")
            _args["tag"] = f"{tag}-{_tag}"
            _nodes = _args.get("nodes", None)
            _tasks = _args.get("tasks_per_node", 8)
            eval_nodes = _nodes if eval_nodes is None else eval_nodes
            eval_tasks_per_node = _tasks if eval_tasks_per_node is None else eval_tasks_per_node
            if (eval_nodes != _nodes) or (eval_tasks_per_node != _tasks):
                warnings.warn("Configs for online evals must use same number of nodes for slurm-batch processing")

            # Model params
            _args["pretrain"] = {}
            _args["pretrain"]["model_name"] = model_name
            _args["pretrain"]["patch_size"] = patch_size
            _args["pretrain"]["tubelet_size"] = tubelet_size
            _args["pretrain"]["uniform_power"] = uniform_power
            _args["pretrain"]["use_sdpa"] = use_sdpa
            _args["pretrain"]["clip_duration"] = clip_duration
            _args["pretrain"]["use_silu"] = use_silu
            _args["pretrain"]["wide_silu"] = wide_silu

            # Data params
            _args["pretrain"]["frames_per_clip"] = num_frames

            # Misc
            _args["pretrain"]["folder"] = logging_folder
            _args["pretrain"]["checkpoint"] = checkpoint
            _args["pretrain"]["write_tag"] = write_tag

            args_eval += [_args]

    return eval_nodes, eval_tasks_per_node, args_eval


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    a2v_predictor,
    a2av_predictor,
    v2a_predictor,
    v2av_predictor,
    av2a_predictor,
    av2v_predictor,
    opt,
    scaler,
    is_anneal=False,
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    # When annealing, start from epoch 0 (fresh LR schedule)
    # Otherwise, restore epoch from checkpoint
    epoch = 0
    if not is_anneal:
        epoch = checkpoint["epoch"]

    # -- loading encoder
    pretrained_dict = checkpoint["encoder"]
    if is_anneal:
        # During annealing/cooldown: handle shape mismatches (e.g., pos_embed)
        # by keeping the model's freshly initialized values
        for k, v in encoder.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                pretrained_dict[k] = v
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
    else:
        msg = encoder.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    # -- loading predictor
    pretrained_dict = checkpoint["predictor"]
    if is_anneal:
        for k, v in predictor.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                pretrained_dict[k] = v
        msg = predictor.load_state_dict(pretrained_dict, strict=False)
    else:
        msg = predictor.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # -- loading video_target_encoder
    if target_encoder is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint["target_encoder"]
        if is_anneal:
            for k, v in target_encoder.state_dict().items():
                if k not in pretrained_dict:
                    logger.info(f'key "{k}" could not be found in loaded state dict')
                elif pretrained_dict[k].shape != v.shape:
                    logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                    pretrained_dict[k] = v
            msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        else:
            msg = target_encoder.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained target_encoder from epoch {epoch} with msg: {msg}")

    # -- loading a2v_predictor
    pretrained_dict = checkpoint["a2v_predictor"]
    msg = a2v_predictor.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained a2v_predictor from epoch {epoch} with msg: {msg}")

    # -- loading a2av_predictor
    pretrained_dict = checkpoint["a2av_predictor"]
    msg = a2av_predictor.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained a2av_predictor from epoch {epoch} with msg: {msg}")

    # -- loading v2a_predictor
    pretrained_dict = checkpoint["v2a_predictor"]
    msg = v2a_predictor.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained v2a_predictor from epoch {epoch} with msg: {msg}")

    # -- loading v2av_predictor
    pretrained_dict = checkpoint["v2av_predictor"]
    msg = v2av_predictor.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained v2av_predictor from epoch {epoch} with msg: {msg}")

    # -- loading av2a_predictor
    pretrained_dict = checkpoint["av2a_predictor"]
    msg = av2a_predictor.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained av2a_predictor from epoch {epoch} with msg: {msg}")

    # -- loading av2v_predictor
    pretrained_dict = checkpoint["av2v_predictor"]
    msg = av2v_predictor.load_state_dict(pretrained_dict)
    logger.info(f"loaded pretrained av2v_predictor from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    opt.load_state_dict(checkpoint["opt"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        a2v_predictor,
        a2av_predictor,
        v2a_predictor,
        v2av_predictor,
        av2a_predictor,
        av2v_predictor,
        opt,
        scaler,
        epoch,
    )


def init_audio_video_model(
    device,
    # video params
    video_patch_size=16,
    video_max_num_frames=16,
    video_tubelet_size=2,
    video_img_size=(224, 224),
    # audio params
    audio_patch_size=16,
    audio_img_size=[1024, 128],
    audio_in_chans=1,
    # model params
    model_name="vit_base",
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    use_activation_checkpointing=False,
    audio_feature_pooling="mean",
    video_feature_pooling="mean",
    av_feature_pooling="mean",
):
    encoder = mjepa_enc.__dict__[model_name](
        video_img_size=video_img_size,
        video_patch_size=video_patch_size,
        video_num_frames=video_max_num_frames,
        tubelet_size=video_tubelet_size,
        audio_img_size=audio_img_size,
        audio_patch_size=audio_patch_size,
        audio_in_chans=audio_in_chans,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )
    predictor = mjepa_pred.__dict__["vit_predictor"](
        video_img_size=video_img_size,
        video_patch_size=video_patch_size,
        video_num_frames=video_max_num_frames,
        video_tubelet_size=video_tubelet_size,
        audio_img_size=audio_img_size,
        audio_patch_size=audio_patch_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        use_mask_tokens=use_mask_tokens,
        num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    a2v_predictor = a2v2predictor(
        mod1_feature_dim=encoder.embed_dim,
        mod2_feature_dim=encoder.embed_dim,
        pooling=audio_feature_pooling,
    )

    a2av_predictor = a2v2predictor(
        mod1_feature_dim=encoder.embed_dim,
        mod2_feature_dim=encoder.embed_dim,
        pooling=audio_feature_pooling,
    )

    v2a_predictor = a2v2predictor(
        mod1_feature_dim=encoder.embed_dim,
        mod2_feature_dim=encoder.embed_dim,
        pooling=video_feature_pooling,
    )

    v2av_predictor = a2v2predictor(
        mod1_feature_dim=encoder.embed_dim,
        mod2_feature_dim=encoder.embed_dim,
        pooling=video_feature_pooling,
    )

    av2a_predictor = a2v2predictor(
        mod1_feature_dim=encoder.embed_dim,
        mod2_feature_dim=encoder.embed_dim,
        pooling=av_feature_pooling,
    )

    av2v_predictor = a2v2predictor(
        mod1_feature_dim=encoder.embed_dim,
        mod2_feature_dim=encoder.embed_dim,
        pooling=av_feature_pooling,
    )

    encoder = MJEPAMultiSeqWrapper(encoder)
    predictor = MJEPAPredictorHierMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"video_encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"video_predictor number of parameters: {count_parameters(predictor)}")

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in a2v_predictor.modules():
        init_weights(m)
    for m in a2av_predictor.modules():
        init_weights(m)
    for m in v2a_predictor.modules():
        init_weights(m)
    for m in v2av_predictor.modules():
        init_weights(m)
    for m in av2a_predictor.modules():
        init_weights(m)
    for m in av2v_predictor.modules():
        init_weights(m)

    a2v_predictor.to(device)
    a2av_predictor.to(device)
    v2a_predictor.to(device)
    v2av_predictor.to(device)
    av2a_predictor.to(device)
    av2v_predictor.to(device)

    logger.info(a2v_predictor)
    logger.info(a2av_predictor)
    logger.info(v2a_predictor)
    logger.info(v2av_predictor)
    logger.info(av2a_predictor)
    logger.info(av2v_predictor)

    return (
        encoder,
        predictor,
        a2v_predictor,
        a2av_predictor,
        v2a_predictor,
        v2av_predictor,
        av2a_predictor,
        av2v_predictor,
    )


def init_opt(
    encoder,
    predictor,
    a2v_predictor,
    a2av_predictor,
    v2a_predictor,
    v2av_predictor,
    av2a_predictor,
    av2v_predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    is_anneal=False,
):
    param_groups = [
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "mod_type": "video",
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "mod_type": "video",
        },
        {
            "params": (p for n, p in a2v_predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "mod_type": "audio",
        },
        {
            "params": (p for n, p in a2av_predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "mod_type": "audio",
        },
        {
            "params": (p for n, p in v2a_predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "mod_type": "video",
        },
        {
            "params": (p for n, p in v2av_predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "mod_type": "video",
        },
        {
            "params": (p for n, p in av2a_predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "mod_type": "video",
        },
        {
            "params": (p for n, p in av2v_predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "mod_type": "video",
        },
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "mod_type": "video",
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "mod_type": "video",
        },
        {
            "params": (p for n, p in a2v_predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "mod_type": "audio",
        },
        {
            "params": (p for n, p in a2av_predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "mod_type": "audio",
        },
        {
            "params": (p for n, p in v2a_predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "mod_type": "video",
        },
        {
            "params": (p for n, p in v2av_predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "mod_type": "video",
        },
        {
            "params": (p for n, p in av2a_predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "mod_type": "video",
        },
        {
            "params": (p for n, p in av2v_predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "mod_type": "video",
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    # Use LinearDecaySchedule for annealing (no warmup, linear decay to final_lr)
    # Use WarmupCosineSchedule for normal pre-training
    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )

    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
