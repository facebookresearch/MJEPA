# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
MJEPA Audio-Video Encoder wrapper for classification evaluation (audio+video features).

This module wraps MJEPA encoder for audio-video joint evaluation on datasets like AudioSet.
It processes both audio spectrogram and video frames through the MJEPA encoder,
and concatenates the audio and video features along the sequence dimension.

API requirements for Encoder module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param audio_x: (Tensor) Audio spectrogram (shape=[B x 1 x T x F])
        :param video_x: (Tensor) Video frames (shape=[B x C x T x H x W])
        :returns: (Tensor) Concatenated audio+video representations
                  (shape=[B x (audio_tokens + video_tokens) x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.
"""

import logging

import torch
import torch.nn as nn

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    """
    Initialize MJEPA encoder for audio-video joint classification.

    Detects the MJEPA encoder configuration from the checkpoint keys:
    - with modality embeddings (`audio_mod_embed` present)
    - without modality embeddings (base encoder)

    Checkpoints from encoder variants not included in this release raise a
    ValueError rather than silently loading the wrong architecture.

    Args:
        checkpoint: Path to pretrained checkpoint
        model_kwargs: Model configuration including encoder params
        wrapper_kwargs: Wrapper configuration (feature_pooling, etc.)

    Returns:
        MJEPAAudioVideoEncoder wrapped model
    """
    logger.info(f"Loading pretrained MJEPA model from {checkpoint}")
    checkpoint_data = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs.get("encoder", {})
    enc_ckp_key = enc_kwargs.get("checkpoint_key", "target_encoder")
    enc_model_name = enc_kwargs.get("model_name", "vit_large")

    # For audio-video, checkpoint must have 'target_encoder' (shared encoder)
    if enc_ckp_key in checkpoint_data:
        pretrained_dict = checkpoint_data[enc_ckp_key]
    elif "target_encoder" in checkpoint_data:
        pretrained_dict = checkpoint_data["target_encoder"]
        logger.info("Using target_encoder key")
    else:
        raise KeyError(
            f"Could not find encoder weights in checkpoint. "
            f"For audio-video input, checkpoint must have 'target_encoder'. "
            f"Available keys: {checkpoint_data.keys()}"
        )

    # Clean up state dict keys
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_key = new_key.replace("backbone.", "") if new_key.startswith("backbone.") else new_key
        new_state_dict[new_key] = v

    # Model params
    video_img_size = enc_kwargs.get("video_img_size", [224, 224])
    video_patch_size = enc_kwargs.get("video_patch_size", 16)
    video_num_frames = enc_kwargs.get("video_num_frames", 16)
    video_tubelet_size = enc_kwargs.get("video_tubelet_size", 2)
    audio_spectrogram_size = enc_kwargs.get("audio_spectrogram_size", [1024, 128])
    audio_patch_size = enc_kwargs.get("audio_patch_size", 16)

    # Common model kwargs for all MJEPA variants
    common_kwargs = dict(
        video_img_size=video_img_size,
        video_patch_size=video_patch_size,
        video_num_frames=video_num_frames,
        video_tubelet_size=video_tubelet_size,
        video_in_chans=3,
        uniform_power=True,
        # audio params
        audio_img_size=audio_spectrogram_size,
        audio_in_chans=1,
        audio_patch_size=audio_patch_size,
        # common params
        use_sdpa=True,
        use_silu=False,
        wide_silu=False,
        use_activation_checkpointing=True,
        use_rope=False,
    )

    # Detect encoder variant based on checkpoint keys and instantiate appropriate model
    if "norms_block.0.weight" in new_state_dict and "audio_mod_embed" in new_state_dict:
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
        logger.info(f"Detected base mjepa_encoder variant, using model: {enc_model_name}")
        import src.models.mjepa_encoder as mjepa_encoder

        common_kwargs_base = dict(common_kwargs)
        common_kwargs_base["audio_spectrogram_size"] = common_kwargs_base.pop("audio_img_size")
        model = mjepa_encoder.__dict__[enc_model_name](**common_kwargs_base)

    # Load pretrained weights with shape mismatch handling
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

    # Wrap model for audio-video processing
    feature_pooling = wrapper_kwargs.get("feature_pooling", "mean")
    wrapped_model = MJEPAAudioVideoEncoder(
        model,
        feature_pooling=feature_pooling,
    )
    del checkpoint_data
    return wrapped_model


class MJEPAAudioVideoEncoder(nn.Module):
    """
    Process audio spectrogram and video frames through MJEPA encoder jointly.

    This wrapper calls the MJEPA encoder with both audio and video inputs,
    then concatenates the audio and video features along the sequence dimension.
    """

    def __init__(
        self,
        model,
        feature_pooling="mean",
    ):
        """
        Initialize MJEPAAudioVideoEncoder.

        Args:
            model: MJEPA encoder model
            feature_pooling: Feature pooling strategy ('mean', 'cls', or 'none')
                - 'mean': Mean pool all concatenated tokens
                - 'cls': Use first token from concatenated features
                - 'none': Return all tokens (for attentive pooler)
        """
        super().__init__()
        self.model = model
        self.embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.feature_pooling = feature_pooling

    def forward(self, audio_x, video_x):
        """
        Forward pass for audio-video joint classification.

        Args:
            audio_x: Audio spectrogram tensor of shape (B, 1, T, F)
                     where T=target_length, F=num_mel_bins
            video_x: Video tensor of shape (B, C, T, H, W)

        Returns:
            Concatenated audio+video features of shape (B, N_a + N_v, D)
            If feature_pooling is 'mean' or 'cls', returns (B, D)
        """
        # Forward through MJEPA encoder with both modalities
        # MJEPA returns (audio_out, video_out) tuple
        audio_features, video_features = self.model(audio_x=audio_x, video_x=video_x)

        # Concatenate audio and video features along the sequence dimension
        # audio_features: (B, N_audio, D), video_features: (B, N_video, D)
        all_features = torch.cat([audio_features, video_features], dim=1)

        if self.feature_pooling == "none":
            return all_features
        elif self.feature_pooling == "cls":
            return all_features[:, 0, :]
        elif self.feature_pooling == "mean":
            return all_features.mean(dim=1)
        else:
            return all_features
