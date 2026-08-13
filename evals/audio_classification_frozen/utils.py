# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for audio classification evaluation.

This module provides audio preprocessing utilities for AudioSet evaluation.
"""

import torch
import torchaudio


def preprocess_audio(
    path,
    target_length=1024,
    norm_mean=-4.268,
    norm_std=4.569,
    sample_rate=16000,
    num_mel_bins=128,
):
    """
    Preprocess audio file to mel spectrogram.

    Args:
        path: Path to audio file
        target_length: Target length for mel spectrogram (default 1024 for 10s audio)
        norm_mean: Normalization mean
        norm_std: Normalization std
        sample_rate: Target sample rate
        num_mel_bins: Number of mel bins

    Returns:
        Tensor of shape [1, target_length, num_mel_bins]
    """
    wav, sr = torchaudio.load(path)

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav - wav.mean()

    mel = torchaudio.compliance.kaldi.fbank(
        wav,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_shift=10,
    ).unsqueeze(0)

    # Pad or truncate to target length
    n_frames = mel.shape[1]
    if n_frames < target_length:
        mel = torch.nn.ZeroPad2d((0, 0, 0, target_length - n_frames))(mel)
    elif n_frames > target_length:
        mel = mel[:, :target_length, :]

    # Normalize
    mel = (mel - norm_mean) / (norm_std * 2)

    return mel  # shape [1, target_length, num_mel_bins]


class AudioTransform:
    """
    Audio transform for evaluation.

    Handles loading and preprocessing audio data.
    """

    def __init__(
        self,
        target_length=1024,
        num_mel_bins=128,
        sample_rate=16000,
        norm_mean=-4.268,
        norm_std=4.569,
    ):
        self.target_length = target_length
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, audio_path):
        """
        Transform audio file to mel spectrogram.

        Args:
            audio_path: Path to audio file

        Returns:
            Tensor of shape [1, target_length, num_mel_bins]
        """
        return preprocess_audio(
            audio_path,
            target_length=self.target_length,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            sample_rate=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
        )


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.

    Args:
        tensor: Tensor to normalize
        mean: Mean value to subtract
        std: Std to divide

    Returns:
        Normalized tensor
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean)
    if isinstance(std, (list, tuple)):
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
