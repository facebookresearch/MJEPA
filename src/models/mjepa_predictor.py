# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from src.masks.utils import apply_masks
from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import repeat_interleave_batch, trunc_normal_


class VisionTransformerPredictor(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        # video params
        video_img_size=(224, 224),
        video_patch_size=16,
        video_num_frames=1,
        video_tubelet_size=2,
        # audio params
        audio_img_size=[1024, 128],
        audio_patch_size=16,
        # common ones
        num_mask_tokens=1,  # we are always using one mask token
        # transformer params
        embed_dim=768,
        predictor_embed_dim=384,
        out_embed_dim=None,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        zero_init_mask_tokens=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        return_all_tokens=False,
        use_rope=False,  # we are not using rope
        # new params for hier
        modality_embedding=True,  # This is True for V-JEPA 2.5 | also adopted for mjepa
        num_hier_layers=4,  # hard coded for now
        **kwargs,
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.use_activation_checkpointing = use_activation_checkpointing
        self.num_hier_layers = num_hier_layers

        # Fusion MLP
        act_layer_mlp = nn.SiLU if use_silu else nn.GELU
        self.predictor_embed = nn.Sequential(
            nn.Linear(embed_dim * self.num_hier_layers, embed_dim, bias=True),
            act_layer_mlp(),
            nn.Linear(embed_dim, predictor_embed_dim, bias=True),
        )

        # Mask tokens
        self.num_mask_tokens = 0
        self.num_mask_tokens = num_mask_tokens  # we use only one kind of mask
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for i in range(num_mask_tokens)]
        )

        # both audio and video mod embedding
        self.modality_embedding = False
        if modality_embedding:
            self.audio_mod_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            nn.init.normal_(self.audio_mod_embed, std=1e-6)
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            self.modality_embedding = True

        # audio
        ##########################################

        # Determine positional embedding
        self.audio_img_size = audio_img_size
        self.audio_patch_size = audio_patch_size
        self.audio_num_patches = (self.audio_img_size[0] // self.audio_patch_size) * (
            self.audio_img_size[1] // self.audio_patch_size
        )
        self.audio_patches_grid_size = [
            (self.audio_img_size[0] // self.audio_patch_size),
            (self.audio_img_size[1] // self.audio_patch_size),
        ]

        self.audio_predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.audio_num_patches, predictor_embed_dim), requires_grad=False
        )
        # init weights
        audio_predictor_pos_embed = audio_get_2d_sincos_pos_embed(
            self.audio_predictor_pos_embed.shape[-1], self.audio_patches_grid_size, cls_token=False
        )
        self.audio_predictor_pos_embed.data.copy_(torch.from_numpy(audio_predictor_pos_embed).float().unsqueeze(0))
        ##########################################

        # video
        ##########################################
        # Determine positional embedding
        if type(video_img_size) is int:
            video_img_size = (video_img_size, video_img_size)
        self.video_img_height, self.video_img_width = video_img_size
        self.video_patch_size = video_patch_size
        self.video_num_frames = video_num_frames
        self.video_tubelet_size = video_tubelet_size
        self.is_video = video_num_frames > 1

        self.video_grid_height = video_img_size[0] // self.video_patch_size
        self.video_grid_width = video_img_size[1] // self.video_patch_size
        self.video_grid_depth = video_num_frames // self.video_tubelet_size

        if self.is_video:
            self.video_num_patches = (
                (video_num_frames // video_tubelet_size)
                * (video_img_size[0] // video_patch_size)
                * (video_img_size[1] // video_patch_size)
            )
        else:
            self.video_num_patches = (video_img_size[0] // video_patch_size) * (video_img_size[1] // video_patch_size)

        # Position embedding
        self.uniform_power = uniform_power
        self.video_predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.video_num_patches, predictor_embed_dim), requires_grad=False
        )

        # init weights
        self.video_init_pos_embed(self.video_predictor_pos_embed.data)  # sincos pos-embed
        ##########################################

        # Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=False,
                    grid_size=0,  # not using this. only used for rope
                    grid_depth=None,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        if out_embed_dim is None:
            out_embed_dim = embed_dim
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(
            predictor_embed_dim, self.num_hier_layers * out_embed_dim, bias=True
        )  # MOD, now we output a single feat map
        if self.return_all_tokens:
            self.predictor_proj_context = nn.Linear(
                predictor_embed_dim, out_embed_dim * self.num_hier_layers, bias=True
            )

        # ------ initialize weights
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def video_init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.video_img_height // self.video_patch_size  # TODO: update; currently assumes square input
        if self.is_video:
            grid_depth = self.video_num_frames // self.video_tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(
        self,
        audio_x=None,
        audio_masks_x=None,
        audio_masks_y=None,
        video_x=None,
        video_masks_x=None,
        video_masks_y=None,
        mask_index=1,
    ):
        """
        :param audio_x: audio_context tokens
        :param audio_masks_x: indices of context tokens in input
        :params audio_masks_y: indices of target tokens in input
        :param video_x: video_context tokens
        :param video_masks_x: indices of context tokens in input
        :params video_masks_y: indices of target tokens in input
        """

        if video_x is None and audio_x is None:
            assert False, "both video and audio can not be none"

        # Bounds check: mask indices must be within the token grid
        if audio_x is not None and audio_masks_x is not None:
            _masks = audio_masks_x if isinstance(audio_masks_x, list) else [audio_masks_x]
            for _m in _masks:
                if _m.max() >= self.audio_num_patches:
                    raise ValueError(f"audio mask index {_m.max()} >= audio_num_patches {self.audio_num_patches}")
        if audio_masks_y is not None:
            _masks = audio_masks_y if isinstance(audio_masks_y, list) else [audio_masks_y]
            for _m in _masks:
                if _m.max() >= self.audio_num_patches:
                    raise ValueError(f"audio mask index {_m.max()} >= audio_num_patches {self.audio_num_patches}")
        if video_x is not None and video_masks_x is not None:
            _masks = video_masks_x if isinstance(video_masks_x, list) else [video_masks_x]
            for _m in _masks:
                if _m.max() >= self.video_num_patches:
                    raise ValueError(f"video mask index {_m.max()} >= video_num_patches {self.video_num_patches}")
        if video_masks_y is not None:
            _masks = video_masks_y if isinstance(video_masks_y, list) else [video_masks_y]
            for _m in _masks:
                if _m.max() >= self.video_num_patches:
                    raise ValueError(f"video mask index {_m.max()} >= video_num_patches {self.video_num_patches}")

        has_audio = False
        has_video = False

        if audio_x is not None:
            has_audio = True

            assert (audio_masks_x is not None) and (
                audio_masks_y is not None
            ), "Cannot run predictor without mask indices"
            if not isinstance(audio_masks_x, list):
                audio_masks_x = [audio_masks_x]
            if not isinstance(audio_masks_y, list):
                audio_masks_y = [audio_masks_y]

            # Batch Size
            audio_B = len(audio_x) // len(audio_masks_x)

            # Map context tokens to pedictor dimensions
            audio_x = self.predictor_embed(audio_x)
            _, audio_N_ctxt, D = audio_x.shape

            # Add positional embedding to ctxt tokens
            audio_x_pos_embed = self.audio_predictor_pos_embed.repeat(audio_B, 1, 1)
            audio_x += apply_masks(audio_x_pos_embed, audio_masks_x)

            # Make target tokens
            mask_index = mask_index % self.num_mask_tokens
            audio_pred_tokens = self.mask_tokens[mask_index]
            audio_pred_tokens = audio_pred_tokens.repeat(audio_B, self.audio_num_patches, 1)
            audio_pred_tokens = apply_masks(audio_pred_tokens, audio_masks_y)

            # -- add pos embed to pred tokens
            audio_pos_embs = self.audio_predictor_pos_embed.repeat(audio_B, 1, 1)
            audio_pos_embs = apply_masks(audio_pos_embs, audio_masks_y)
            audio_pos_embs = repeat_interleave_batch(audio_pos_embs, audio_B, repeat=len(audio_masks_x))
            audio_pred_tokens += audio_pos_embs

            _, audio_pred_N, _ = audio_pred_tokens.shape

            # Concatenate context & target tokens
            audio_x = audio_x.repeat(len(audio_masks_x), 1, 1)
            audio_x = torch.cat([audio_x, audio_pred_tokens], dim=1)

            # add type embedding to all tokens (context + pred)
            audio_x = audio_x + self.audio_mod_embed.repeat(audio_B, 1, 1)

            _, audio_N_total, D = audio_x.shape  # after adding both contex and pred tokens

            # No token re-sorting is needed here because RoPE is not used.

        if video_x is not None:
            has_video = True

            assert (video_masks_x is not None) and (
                video_masks_y is not None
            ), "Cannot run predictor without mask indices"
            if not isinstance(video_masks_x, list):
                video_masks_x = [video_masks_x]
            if not isinstance(video_masks_y, list):
                video_masks_y = [video_masks_y]

            # Batch Size
            video_B = len(video_x) // len(video_masks_x)

            # Map context tokens to pedictor dimensions
            video_x = self.predictor_embed(video_x)
            _, video_N_ctxt, D = video_x.shape

            # Add positional embedding to ctxt tokens
            video_x_pos_embed = self.video_predictor_pos_embed.repeat(video_B, 1, 1)
            video_x += apply_masks(video_x_pos_embed, video_masks_x)

            # Make target tokens
            mask_index = mask_index % self.num_mask_tokens
            video_pred_tokens = self.mask_tokens[mask_index]
            video_pred_tokens = video_pred_tokens.repeat(video_B, self.video_num_patches, 1)
            video_pred_tokens = apply_masks(video_pred_tokens, video_masks_y)

            # -- add pos embed to pred tokens
            video_pos_embs = self.video_predictor_pos_embed.repeat(video_B, 1, 1)
            video_pos_embs = apply_masks(video_pos_embs, video_masks_y)
            video_pos_embs = repeat_interleave_batch(video_pos_embs, video_B, repeat=len(video_masks_x))
            video_pred_tokens += video_pos_embs

            _, video_pred_N, _ = video_pred_tokens.shape

            # Concatenate context & target tokens
            video_x = video_x.repeat(len(video_masks_x), 1, 1)
            video_x = torch.cat([video_x, video_pred_tokens], dim=1)

            # add type embedding to all vision tokens (context + pred)
            video_x = video_x + self.video_mod_embed.repeat(video_B, 1, 1)

            _, video_N_total, D = video_x.shape  # after adding both contex and pred tokens

            # No token re-sorting is needed here because RoPE is not used.

        # Fwd prop
        if has_audio and has_video:
            x = torch.cat([audio_x, video_x], dim=1)
        elif has_audio and not has_video:
            x = audio_x
        elif not has_audio and has_video:
            x = video_x
        else:
            assert False, "at least one of audio/video should be there"

        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, mask=None, attn_mask=None, use_reentrant=False)
            else:
                x = blk(x, mask=None, attn_mask=None)
        x = self.predictor_norm(x)

        # out now

        # Return output corresponding to target tokens
        if not self.return_all_tokens:
            audio_pred = None
            video_pred = None

            if has_audio and has_video:
                # separating audio and video tokens
                audio_out = x[:, :audio_N_total, :]
                video_out = x[:, audio_N_total : audio_N_total + video_N_total, :]

                audio_pred = audio_out[:, audio_N_ctxt:]  # only pred tokens
                video_pred = video_out[:, video_N_ctxt:]  # only pred tokens

                audio_pred = self.predictor_proj(audio_pred)
                video_pred = self.predictor_proj(video_pred)

                assert audio_pred.size(1) == audio_pred_N
                assert video_pred.size(1) == video_pred_N
            elif has_audio and not has_video:
                audio_out = x
                audio_pred = audio_out[:, audio_N_ctxt:]  # only pred tokens
                audio_pred = self.predictor_proj(audio_pred)
                assert audio_pred.size(1) == audio_pred_N
            elif not has_audio and has_video:
                video_out = x
                video_pred = video_out[:, video_N_ctxt:]  # only pred tokens
                video_pred = self.predictor_proj(video_pred)
                assert video_pred.size(1) == video_pred_N
            else:
                assert False, "at least one of audio/video should be there"

            return audio_pred, video_pred, None, None  # for context tokens

        elif self.return_all_tokens:
            audio_pred = None
            audio_context = None
            video_pred = None
            video_context = None

            if has_audio and has_video:
                # separating audio and video tokens
                audio_out = x[:, :audio_N_total, :]
                video_out = x[:, audio_N_total : audio_N_total + video_N_total, :]

                audio_pred = audio_out[:, audio_N_ctxt:]  # only pred tokens
                video_pred = video_out[:, video_N_ctxt:]  # only pred tokens

                audio_context = audio_out[:, :audio_N_ctxt]  # only context tokens
                video_context = video_out[:, :video_N_ctxt]  # only context tokens

                audio_pred = self.predictor_proj(audio_pred)
                video_pred = self.predictor_proj(video_pred)

                audio_context = self.predictor_proj_context(audio_context)
                video_context = self.predictor_proj_context(video_context)

                assert audio_pred.size(1) == audio_pred_N
                assert video_pred.size(1) == video_pred_N
            elif has_audio and not has_video:
                audio_out = x
                audio_pred = audio_out[:, audio_N_ctxt:]  # only pred tokens
                audio_context = audio_out[:, :audio_N_ctxt]  # only context tokens

                audio_pred = self.predictor_proj(audio_pred)
                audio_context = self.predictor_proj_context(audio_context)

                assert audio_pred.size(1) == audio_pred_N
            elif not has_audio and has_video:
                video_out = x
                video_pred = video_out[:, video_N_ctxt:]  # only pred tokens
                video_context = video_out[:, :video_N_ctxt]  # only context tokens

                video_pred = self.predictor_proj(video_pred)
                video_context = self.predictor_proj_context(video_context)

                assert video_pred.size(1) == video_pred_N
            else:
                assert False, "at least one of audio/video should be there"

            return audio_pred, video_pred, audio_context, video_context  # for context tokens


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


# audio pos embedding
#############################
def audio_get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: [h, w] of tokens
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = audio_get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def audio_get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = audio_get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = audio_get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def audio_get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = audio_get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def audio_get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#############################
