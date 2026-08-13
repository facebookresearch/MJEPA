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
from src.models.utils.patch_embed import PatchEmbed, PatchEmbed3D
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import trunc_normal_


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        # video params
        video_img_size=(224, 224),
        video_patch_size=16,
        video_num_frames=1,
        video_tubelet_size=2,
        video_in_chans=3,
        # audio params
        audio_img_size=[1024, 128],
        audio_in_chans=1,
        audio_patch_size=16,
        # transformer params
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        use_rope=False,
        handle_nonsquare_inputs=True,
        # new params for hier
        modality_embedding=True,  # This is True for V-JEPA 2.5 | also adopted for mjepa
        n_output_distillation=4,  # For deciding if we want to use hierarchical loss during distillation
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.use_activation_checkpointing = use_activation_checkpointing

        # video
        ##########################################
        if type(video_img_size) is int:
            video_img_size = (video_img_size, video_img_size)
        self.video_img_height, self.video_img_width = video_img_size
        self.video_patch_size = video_patch_size
        self.video_num_frames = video_num_frames
        self.video_tubelet_size = video_tubelet_size
        self.is_video = video_num_frames > 1
        # self.handle_nonsquare_inputs = handle_nonsquare_inputs  # note use wo rope

        # Tokenize pixels with convolution
        if self.is_video:
            self.video_patch_embed = PatchEmbed3D(
                patch_size=video_patch_size,
                tubelet_size=video_tubelet_size,
                in_chans=video_in_chans,
                embed_dim=embed_dim,
            )
            num_patches_t = video_num_frames // video_tubelet_size
            num_patches_h = video_img_size[0] // video_patch_size
            num_patches_w = video_img_size[1] // video_patch_size
            self.video_num_patches = num_patches_t * num_patches_h * num_patches_w
        else:
            self.video_patch_embed = PatchEmbed(
                patch_size=video_patch_size, in_chans=video_in_chans, embed_dim=embed_dim
            )
            num_patches_h = video_img_size[0] // video_patch_size
            num_patches_w = video_img_size[1] // video_patch_size
            self.video_num_patches = num_patches_h * num_patches_w

        # Position embedding
        self.uniform_power = uniform_power
        self.video_pos_embed = nn.Parameter(torch.zeros(1, self.video_num_patches, embed_dim), requires_grad=False)

        # NOTE: NOT ADDING ANY TYPE EMBEDDING: as that will be captured by patch emb bias
        ##########################################

        # audio
        ##########################################
        self.audio_img_size = audio_img_size
        self.audio_in_chans = audio_in_chans
        self.audio_patch_size = audio_patch_size

        # patch token embedding
        self.audio_patch_embed = AudioPatchEmbed(
            img_size=audio_img_size, patch_size=audio_patch_size, in_chans=audio_in_chans, embed_dim=embed_dim
        )
        self.audio_num_patches = self.audio_patch_embed.num_patches

        # position embedding
        self.audio_pos_embed = nn.Parameter(torch.zeros(1, self.audio_num_patches, embed_dim), requires_grad=False)
        audio_pos_embed = audio_get_2d_sincos_pos_embed(
            self.audio_pos_embed.shape[-1], self.audio_patch_embed.patches_grid_size, cls_token=False
        )
        self.audio_pos_embed.data.copy_(torch.from_numpy(audio_pos_embed).float().unsqueeze(0))
        ##########################################

        # Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=False,
                    grid_size=0,  # not used if no rope
                    grid_depth=None,  # not used if no rope
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
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

        # ------ initialize weights
        if self.video_pos_embed is not None:
            self._init_video_pos_embed(self.video_pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        # self.norm = norm_layer(embed_dim)  # replaced by a whole norm block for each hier layer

        # MOD FOR HIERARCHICAL LOSS
        if depth == 12:
            self.hierarchical_layers = [2, 5, 8, 11]
            if n_output_distillation == 4:
                self.out_layers_distillation = [2, 5, 8, 11]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [11]

        elif depth == 24:
            self.hierarchical_layers = [5, 11, 17, 23]
            if n_output_distillation == 4:
                self.out_layers_distillation = [5, 11, 17, 23]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [23]

        elif depth == 32:
            self.hierarchical_layers = [7, 15, 23, 31]
            if n_output_distillation == 4:
                self.out_layers_distillation = [7, 15, 23, 31]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [31]

        elif depth == 40:
            self.hierarchical_layers = [9, 19, 29, 39]
            if n_output_distillation == 4:
                self.out_layers_distillation = [9, 19, 29, 39]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [39]

        elif depth == 48:
            self.hierarchical_layers = [11, 23, 37, 47]  # [36, 37, 38, 39]
            if n_output_distillation == 4:
                self.out_layers_distillation = [11, 23, 37, 47]
            elif n_output_distillation == 1:
                self.out_layers_distillation = [47]
        else:
            raise ValueError(
                f"Unsupported depth={depth}. Supported depths are: 12, 24, 32, 40, 48. "
                f"Please add hierarchical_layers configuration for depth={depth}."
            )
        self.norms_block = nn.ModuleList([norm_layer(embed_dim) for _ in range(len(self.hierarchical_layers))])

        # Mod for multi-modal pos embeddings
        self.modality_embedding = False
        if modality_embedding:
            self.audio_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.audio_mod_embed, std=1e-6)
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            self.modality_embedding = True

    def _init_video_pos_embed(self, pos_embed):
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

        # this layer norm init is adopted from jepa2.5
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            return

        # this is default
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def forward(
        self,
        audio_x=None,
        audio_masks=None,
        video_x=None,
        video_masks=None,
        training=False,
    ):
        """
        :param video_x: input image/video
        :param video_masks: indices of patch tokens to keep
        :param audio_x: input audio spectrogram
        :param audio_masks: indices of patch tokens to keep
        """

        has_audio = False
        has_video = False

        if video_x is None and audio_x is None:
            assert False, "No input to the encoder. Both video and audio are none"

        if audio_x is not None:
            has_audio = True
            # Lets process audio tokens
            ####################################################
            if audio_masks is not None:
                if not isinstance(audio_masks, list):
                    audio_masks = [audio_masks]

            audio_B, audio_C, audio_H, audio_W = audio_x.shape
            # -- patchify audio_x
            audio_x = self.audio_patch_embed(audio_x)
            audio_B, audio_N, audio_D = audio_x.shape

            # -- add positional embedding to audio_x
            audio_input_grid_H = audio_H // self.audio_patch_embed.patch_size
            audio_input_grid_W = audio_W // self.audio_patch_embed.patch_size
            audio_pos_embed = self.audio_interpolate_pos_encoding(
                self.audio_pos_embed, audio_input_grid_H, audio_input_grid_W
            )
            audio_x = audio_x + audio_pos_embed

            # lets add modality type embedding
            audio_x = audio_x + self.audio_mod_embed.repeat(audio_B, 1, 1)

            # -- mask audio_x
            if audio_masks is not None:
                # Bounds check: mask indices must be within the token grid
                for _m in audio_masks:
                    if _m.max() >= audio_N:
                        raise ValueError(f"audio mask index {_m.max()} >= audio_N {audio_N}")
                audio_x = apply_masks(audio_x, audio_masks)

            _, audio_context_N, _ = audio_x.shape
            # the effective batch size will be B * num_masks

            ####################################################

        if video_x is not None:
            has_video = True
            # Lets process video tokens
            ####################################################
            if video_masks is not None and not isinstance(video_masks, list):
                video_masks = [video_masks]

            # Tokenize input
            # Image
            if video_x.ndim == 4:
                _, _, video_H, video_W = video_x.shape
                video_T = 1
            # Video
            elif video_x.ndim == 5:
                _, _, video_T, video_H, video_W = video_x.shape
                video_T = video_T // self.video_tubelet_size
            else:
                assert False, "unknown input dim"

            # get pos embedding first before patchify
            video_pos_embed = self.video_interpolate_pos_encoding(video_x, self.video_pos_embed)

            # patchify video into tokens
            video_x = self.video_patch_embed(video_x)
            video_B, video_N, video_D = video_x.shape

            # -- add positional embedding to video_x
            video_x += video_pos_embed

            # lets add modality type embedding
            video_x = video_x + self.video_mod_embed.repeat(video_B, 1, 1)

            # Mask video_x
            if video_masks is not None:
                # Bounds check: mask indices must be within the token grid
                for _m in video_masks:
                    if _m.max() >= video_N:
                        raise ValueError(f"video mask index {_m.max()} >= video_N {video_N}")
                video_x = apply_masks(video_x, video_masks)

            _, video_context_N, _ = video_x.shape  # context tokens remaining after masking
            # the effective batch size will be B * num_masks

            ####################################################

        # Fwd prop
        if has_audio and has_video:
            x = torch.cat([audio_x, video_x], dim=1)
        elif has_audio and not has_video:
            x = audio_x
        elif not has_audio and has_video:
            x = video_x
        else:
            assert False, "at least one of audio/video should be there"

        # NOTE: since we are not using rope, we dont need to pass masks, T, patches etc to attn block
        outs = []
        hier = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, use_reentrant=False
                )
            else:
                x = blk(x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None)

            # This is used at interence if you want to extract multi-level features.
            if self.out_layers is not None and i in self.out_layers:
                out_idx = self.hierarchical_layers.index(i)
                out_norm = self.norms_block[out_idx](x)
                outs.append(out_norm)

            # THIS IS USED AT TRAINING/DISTILLATION.
            if i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))

        # TODO: expose these multi-level features (`outs`) at inference time

        if training:
            x = torch.cat(hier, dim=-1)  # In training/distillation, we return multi-level features
        else:
            x = self.norms_block[-1](x)  # In inference we only output the last-level

        # out now
        audio_out = None
        video_out = None

        if has_audio and has_video:
            audio_out = x[:, :audio_context_N, :]
            video_out = x[:, audio_context_N : audio_context_N + video_context_N, :]

            assert audio_out.size(1) == audio_context_N
            assert video_out.size(1) == video_context_N
        elif has_audio and not has_video:
            audio_out = x
            assert audio_out.size(1) == audio_context_N
        elif not has_audio and has_video:
            video_out = x
            assert video_out.size(1) == video_context_N
        else:
            assert False, "at least one of audio/video should be there"

        # during training each token is dim*num_hier_layers dimension
        return audio_out, video_out

    def audio_interpolate_pos_encoding(self, pos_embed, input_grid_H, input_grid_W):
        # Example shapes
        train_H, train_W = self.audio_patch_embed.patches_grid_size
        if input_grid_H == train_H and input_grid_W == train_W:
            return pos_embed

        D = pos_embed.shape[-1]  # embedding dimension
        # Reshape to [1, D, H, W] for interpolation
        pos_embed_2d = pos_embed.reshape(1, train_H, train_W, D).permute(0, 3, 1, 2)
        # Interpolate to new grid size
        pos_embed_2d = nn.functional.interpolate(
            pos_embed_2d,
            size=(input_grid_H, input_grid_W),  # input (height, width)
            mode="bicubic",
            align_corners=False,
        )
        # Flatten back to [1, N_new, D]
        pos_embed_new = pos_embed_2d.permute(0, 2, 3, 1).reshape(1, input_grid_H * input_grid_W, D)
        return pos_embed_new

    def video_interpolate_pos_encoding(self, x, pos_embed):

        _, N, dim = pos_embed.shape

        if self.is_video:

            # If pos_embed already corret size, just return
            _, _, T, H, W = x.shape
            if H == self.video_img_height and W == self.video_img_width and T == self.video_num_frames:
                return pos_embed

            # Just chop off last N tokens of positional embedding
            elif H == self.video_img_height and W == self.video_img_width and T < self.video_num_frames:
                new_N = int(
                    (T // self.video_tubelet_size) * (H // self.video_patch_size) * (W // self.video_patch_size)
                )
                return pos_embed[:, :new_N, :]

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.video_tubelet_size
            H = H // self.video_patch_size
            W = W // self.video_patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.video_num_frames // self.video_tubelet_size
            N_h = self.video_img_height // self.video_patch_size
            N_w = self.video_img_width // self.video_patch_size
            assert N_h * N_w * N_t == N, "Positional embedding initialized incorrectly"

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T / N_t, H / N_h, W / N_w)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode="trilinear",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:

            # If pos_embed already corret size, just return
            _, _, H, W = x.shape
            if H == self.video_img_height and W == self.video_img_width:
                return pos_embed

            # Compute scale factor for spatial interpolation
            npatch = (H // self.video_patch_size) * (W // self.video_patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode="bicubic",
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed


###
# audio patch embed


class AudioPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=[1024, 128], patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patches_grid_size = [(img_size[0] // patch_size), (img_size[1] // patch_size)]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# audio pos embedding
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


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=22,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


VIT_EMBED_DIMS = {
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant": 1408,
}
