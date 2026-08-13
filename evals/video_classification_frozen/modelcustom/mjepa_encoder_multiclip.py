# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
MJEPA Video Encoder wrapper for multi-clip video classification evaluation.

This module wraps MJEPA encoder for video-only evaluation on datasets like K400, SSv2.
It handles multiple temporal clips and spatial views, processing them through the
MJEPA encoder and aggregating the outputs.

API requirements for Encoder module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: list of temporal clips, each containing list of spatial views
                  where each view is a Tensor (shape=[batch_size x num_channels x num_frames x height x width])
        :param clip_indices: list of tensors containing frame indices for each clip
        :returns: list of Tensor per spatial view (shape=[batch_size x num_encoder_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.
"""

import logging

import torch
import torch.nn as nn

from src.masks.utils import apply_masks
from src.models.utils.pos_embs import get_1d_sincos_pos_embed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    """
    Initialize MJEPA encoder for video classification.

    Detects the MJEPA encoder configuration from the checkpoint keys:
    - with modality embeddings (`audio_mod_embed` present)
    - without modality embeddings (base encoder)

    Checkpoints from encoder variants not included in this release raise a
    ValueError rather than silently loading the wrong architecture.
    """
    logger.info(f"Loading pretrained MJEPA model from {checkpoint}")
    checkpoint_data = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key", "target_encoder")
    enc_model_name = enc_kwargs.get("model_name", "vit_large")

    # Get pretrained weights
    pretrained_dict = checkpoint_data[enc_ckp_key]

    # Clean up state dict keys
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_key = new_key.replace("backbone.", "") if new_key.startswith("backbone.") else new_key
        new_state_dict[new_key] = v

    # Detect encoder variant based on checkpoint keys and instantiate appropriate model
    # Get encoder-specific kwargs
    video_patch_size = enc_kwargs.get("patch_size", 16)
    video_tubelet_size = enc_kwargs.get("tubelet_size", 2)
    uniform_power = enc_kwargs.get("uniform_power", True)
    use_rope = enc_kwargs.get("use_rope", False)
    video_img_size = enc_kwargs.get("video_img_size", [resolution, resolution])

    # Common model kwargs for all MJEPA variants
    common_kwargs = dict(
        video_img_size=video_img_size,
        video_patch_size=video_patch_size,
        video_num_frames=frames_per_clip,
        video_tubelet_size=video_tubelet_size,
        video_in_chans=3,
        uniform_power=uniform_power,
        # audio params (default, won't be used for video-only eval)
        audio_img_size=[1024, 128],
        audio_in_chans=1,
        audio_patch_size=16,
        # common params
        use_sdpa=True,
        use_silu=False,
        wide_silu=False,
        use_activation_checkpointing=True,
        use_rope=use_rope,
    )

    if "norms_block.0.weight" in new_state_dict and "audio_mod_embed" in new_state_dict:
        # Hierarchical model with modality embeddings (mjepa_encoder)
        logger.info(f"Detected mjepa_encoder variant, using model: {enc_model_name}")
        import src.models.mjepa_encoder as mjepa_encoder

        model = mjepa_encoder.__dict__[enc_model_name](modality_embedding=True, **common_kwargs)
    elif "shared_modality_embedding.weight" in new_state_dict:
        # Encoder variants using a shared modality (type) embedding are not part of
        # this release; only the modality-embedding encoder in
        # src/models/mjepa_encoder.py is included.
        raise ValueError(
            "This checkpoint was trained with a shared modality-embedding encoder "
            "variant, which is not included in this repository. Only checkpoints "
            "produced by the released configs are supported."
        )
    else:
        # Base MJEPA encoder
        logger.info(f"Detected base mjepa_encoder variant, using model: {enc_model_name}")
        import src.models.mjepa_encoder as mjepa_encoder

        model = mjepa_encoder.__dict__[enc_model_name](**common_kwargs)

    # Load pretrained weights with shape mismatch handling
    # (needed for cooldown checkpoints where pos_embed shape differs)
    for k, v in model.state_dict().items():
        if k not in new_state_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif new_state_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            new_state_dict[k] = v
    msg = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"Loaded pretrained MJEPA model with msg: {msg}")
    if "epoch" in checkpoint_data:
        logger.info(f"Loaded from epoch: {checkpoint_data['epoch']}")
    print(model)

    # Wrap model for multi-clip aggregation
    model = MJEPAClipAggregation(
        model,
        tubelet_size=video_tubelet_size,
        **wrapper_kwargs,
    )
    del checkpoint_data
    return model


class MJEPAClipAggregation(nn.Module):
    """
    Process each clip independently through MJEPA encoder (video-only)
    and concatenate all tokens across temporal clips.

    This is similar to ClipAggregation in vit_encoder_multiclip.py but
    adapted for MJEPA's forward signature which returns (audio_out, video_out).
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=128,
        use_pos_embed=False,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads

        # 1D-temporal pos-embedding for aggregating across clips
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):
        """
        Forward pass for multi-clip video classification.

        Args:
            x: list of temporal clips, each containing list of spatial views
               x[temporal_idx][spatial_view_idx] = Tensor(B, C, T, H, W)
            clip_indices: list of tensors with frame indices for each clip

        Returns:
            list of outputs, one per spatial view
            Each output is Tensor(B, N_total_tokens, D)
        """
        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, F, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along batch dimension
        # x[i][j] is the j-th spatial view of the i-th temporal clip
        x_cat = [torch.cat(xi, dim=0) for xi in x]  # list of (B*num_views, C, F, H, W)
        x_cat = torch.cat(x_cat, dim=0)  # (B*num_views*num_clips, C, F, H, W)

        # Forward through MJEPA encoder (video-only)
        # MJEPA returns (audio_out, video_out) tuple
        _, outputs = self.model(video_x=x_cat, audio_x=None)

        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            T = F // self.tubelet_size  # num temporal indices per clip
            S = N // T  # num spatial tokens per temporal index

            # Unroll outputs into a 2D array [spatial_views x temporal_clips]
            eff_B = B * num_views_per_clip
            all_outputs = [[] for _ in range(num_views_per_clip)]
            for i in range(num_clips):
                o = outputs[i * eff_B : (i + 1) * eff_B]
                for j in range(num_views_per_clip):
                    all_outputs[j].append(o[j * B : (j + 1) * B])

            for i, outputs in enumerate(all_outputs):
                # Concatenate along temporal dimension
                outputs = [o.reshape(B, T, S, D) for o in outputs]
                outputs = torch.cat(outputs, dim=1).flatten(1, 2)
                # Compute positional embedding
                if (self.pos_embed is not None) and (clip_indices is not None):
                    _indices = [c[:, :: self.tubelet_size] for c in clip_indices]
                    pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, max_T, D]
                    pos_embed = apply_masks(pos_embed, _indices, concat=False)  # list(Tensor([B, T, D]))
                    pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                    pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, S, 1)  # [B, T*num_clips, S, D]
                    pos_embed = pos_embed.flatten(1, 2)
                    outputs += pos_embed
                all_outputs[i] = outputs

            return all_outputs

        return multiviews_postprocess(outputs)
