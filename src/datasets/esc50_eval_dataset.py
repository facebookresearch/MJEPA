# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
ESC-50 Dataset for Audio Classification Evaluation

ESC-50 is an environmental sound classification dataset with:
- 50 classes of environmental sounds
- 2000 audio clips (40 per class)
- 5-fold cross-validation structure
- 5 second clips at 44.1kHz (resampled to 16kHz)
- Single-label classification

Dataset structure:
- Train/eval JSON files for each fold (esc_train_data_{fold}.json, esc_eval_data_{fold}.json)
- Labels mapping CSV (esc_class_labels_indices.csv)
"""

import csv
import json
import logging
import os

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class ESC50Eval(torch.utils.data.Dataset):
    """
    ESC-50 Dataset for frozen encoder evaluation.

    Args:
        dataset_json: Path to dataset JSON file (train or eval for a specific fold)
        labels_csv: Path to class labels CSV file
        num_mel_bins: Number of mel filterbank bins
        target_length: Target length of mel spectrogram in frames
        sample_rate: Target sample rate for audio
        mean: Normalization mean
        std: Normalization std
        training: Whether in training mode (for augmentation)
    """

    def __init__(
        self,
        dataset_json,
        labels_csv,
        num_mel_bins=128,
        target_length=512,  # 5 seconds at 10ms frame shift
        sample_rate=16000,
        mean=-4.268,
        std=4.569,
        training=False,
    ):
        self.dataset_json = dataset_json
        self.labels_csv = labels_csv
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.mean = mean
        self.std = std
        self.training = training

        # Load label mapping
        self.label_to_idx = self._load_label_mapping()
        self.num_classes = len(self.label_to_idx)

        # Load dataset
        self.samples = self._load_dataset()

        logger.info(
            f"ESC50Eval: Loaded {len(self.samples)} samples, " f"{self.num_classes} classes, training={training}"
        )

    def _load_label_mapping(self):
        """Load label mapping from CSV file."""
        label_to_idx = {}
        with open(self.labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row["mid"]
                idx = int(row["index"])
                label_to_idx[mid] = idx
        return label_to_idx

    def _load_dataset(self):
        """Load dataset from JSON file."""
        with open(self.dataset_json, "r") as f:
            data = json.load(f)["data"]

        samples = []
        for item in data:
            audio_path = item["wav"]
            label_mid = item["labels"]
            label_idx = self.label_to_idx[label_mid]
            samples.append((audio_path, label_idx))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio_path, label_idx = self.samples[index]

        # Try to load audio, skip to random sample if failed
        mel = None
        attempts = 0
        max_attempts = 10

        while mel is None and attempts < max_attempts:
            try:
                mel = self._load_and_transform_audio(audio_path)
            except Exception as e:
                logger.warning(f"Failed to load {audio_path}: {e}")
                # Pick a random sample
                index = np.random.randint(len(self.samples))
                audio_path, label_idx = self.samples[index]
                attempts += 1

        if mel is None:
            raise RuntimeError(f"Failed to load audio after {max_attempts} attempts")

        # Return in format compatible with audio_classification_frozen eval
        # {"audio": tensor} format
        return {"audio": mel}, label_idx

    def _load_and_transform_audio(self, audio_path):
        """Load audio file and convert to mel spectrogram."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        wav, sr = torchaudio.load(audio_path)

        # Resample if necessary
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Convert stereo to mono if necessary
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Normalize waveform
        wav = wav - wav.mean()

        # Compute mel spectrogram using Kaldi-compatible features
        mel = torchaudio.compliance.kaldi.fbank(
            wav,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_shift=10,  # 10ms frame shift
        )

        # mel shape: (T, num_mel_bins)
        n_frames = mel.shape[0]

        # Pad or truncate to target length
        if n_frames < self.target_length:
            # Pad with zeros
            pad_length = self.target_length - n_frames
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_length))
        elif n_frames > self.target_length:
            # Random crop for training, center crop for eval
            if self.training:
                start = np.random.randint(0, n_frames - self.target_length + 1)
            else:
                start = (n_frames - self.target_length) // 2
            mel = mel[start : start + self.target_length, :]

        # Normalize
        mel = (mel - self.mean) / (self.std * 2)

        # Add channel dimension: (T, F) -> (1, T, F)
        mel = mel.unsqueeze(0)

        return mel


class ESC50FoldDataset:
    """
    Helper class to manage ESC-50 5-fold cross-validation.

    Creates train and eval datasets for all 5 folds.
    """

    def __init__(
        self,
        data_dir,
        labels_csv,
        num_mel_bins=128,
        target_length=512,
        sample_rate=16000,
        mean=-4.268,
        std=4.569,
    ):
        """
        Args:
            data_dir: Directory containing esc_train_data_{fold}.json and esc_eval_data_{fold}.json
            labels_csv: Path to class labels CSV file
            num_mel_bins: Number of mel filterbank bins
            target_length: Target length in frames
            sample_rate: Target sample rate
            mean: Normalization mean
            std: Normalization std
        """
        self.data_dir = data_dir
        self.labels_csv = labels_csv
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.mean = mean
        self.std = std

    def get_fold_datasets(self, fold):
        """
        Get train and eval datasets for a specific fold.

        Args:
            fold: Fold number (1-5)

        Returns:
            (train_dataset, eval_dataset)
        """
        train_json = os.path.join(self.data_dir, f"esc_train_data_{fold}.json")
        eval_json = os.path.join(self.data_dir, f"esc_eval_data_{fold}.json")

        train_dataset = ESC50Eval(
            dataset_json=train_json,
            labels_csv=self.labels_csv,
            num_mel_bins=self.num_mel_bins,
            target_length=self.target_length,
            sample_rate=self.sample_rate,
            mean=self.mean,
            std=self.std,
            training=True,
        )

        eval_dataset = ESC50Eval(
            dataset_json=eval_json,
            labels_csv=self.labels_csv,
            num_mel_bins=self.num_mel_bins,
            target_length=self.target_length,
            sample_rate=self.sample_rate,
            mean=self.mean,
            std=self.std,
            training=False,
        )

        return train_dataset, eval_dataset
