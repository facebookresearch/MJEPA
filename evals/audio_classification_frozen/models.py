# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Model initialization module for audio classification evaluation.

This module provides a unified interface for loading and initializing
frozen audio encoders for AudioSet classification evaluation.
"""

import importlib
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add module names here to suppress printing their full architecture, which can
# clutter logs for very large encoders. Matched as a substring of `module_name`.
SKIP_MODEL_PRINT = []


def init_module(
    module_name,
    device,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
):
    """
    Build (frozen) audio encoder and initialize from pretrained checkpoint.

    API requirements for Encoder module:
        1) Needs to be a pytorch module with 'forward()' function protocol:
            :param x: (Tensor) Audio spectrogram (shape=[batch_size x 1 x num_mel_bins x target_length])
            :returns: (Tensor) Representations of audio (shape=[batch_size x num_encoder_tokens x feature_dim])
        2) Needs to have a public attribute called 'embed_dim' (int) describing its
            output feature dimension.

    Args:
        module_name: Full module path (e.g., 'evals.audio_classification_frozen.modelcustom.mjepa_encoder_audio')
        device: Target device for the model
        checkpoint: Path to pretrained checkpoint
        model_kwargs: Model-specific kwargs (encoder config, checkpoint keys, etc.)
        wrapper_kwargs: Wrapper-specific kwargs (feature pooling, etc.)

    Returns:
        Frozen encoder module
    """
    model = (
        importlib.import_module(f"{module_name}")
        .init_module(
            checkpoint=checkpoint,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        .to(device)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Only print the full architecture for models not listed in SKIP_MODEL_PRINT
    should_print = not any(skip_name in module_name for skip_name in SKIP_MODEL_PRINT)
    if should_print:
        print(model)
    else:
        # For large models, just log basic info
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: {module_name.split('.')[-1]} ({num_params/1e6:.1f}M params)")

    return model
