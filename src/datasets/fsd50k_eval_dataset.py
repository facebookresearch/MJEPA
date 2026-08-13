# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
FSD50K Dataset for Audio Classification Evaluation

This module provides a dataset class for FSD50K audio classification
with support for multi-clip evaluation similar to Kinetics-400 video evaluation.

FSD50K is a multi-label audio classification dataset with 200 classes.
Audio clips have variable lengths, so we support:
- Single clip: pad/truncate to target_length
- Multi-clip: extract multiple clips and aggregate predictions

Dataset structure:
    /FSD50K_files/
        dev/                    # Development set audio files (*.wav)
        eval/                   # Evaluation set audio files (*.wav)
        FSD50K_ground_truth/
            dev.csv             # fname,labels,mids,split (split: train/val)
            eval.csv            # fname,labels,mids
            vocabulary.csv      # index,label_name,mid (200 classes)
"""

import csv
import os
from logging import getLogger

import numpy as np
import torch
import torchaudio

_GLOBAL_SEED = 0
logger = getLogger()


class FSD50KEval(torch.utils.data.Dataset):
    """
    FSD50K dataset for audio classification evaluation.

    Supports multi-clip evaluation where multiple clips are extracted from
    each audio sample and predictions are aggregated.
    """

    def __init__(
        self,
        audio_data_dir,
        dataset_csv,
        vocabulary_csv,
        num_mel_bins=128,
        target_length=512,  # 5s at 10ms frame shift
        sample_rate=16000,
        mean=-4.268,
        std=4.569,
        num_clips=1,  # Number of clips to extract per sample
        clip_overlap=True,  # Allow overlapping clips
        training=True,
        debug=False,
    ):
        """
        Initialize FSD50K dataset.

        Args:
            audio_data_dir: Directory containing audio files
            dataset_csv: Path to CSV with labels (dev.csv or eval.csv)
            vocabulary_csv: Path to vocabulary.csv with class indices
            num_mel_bins: Number of mel filterbank bins
            target_length: Target length in mel frames (e.g., 512 for 5s)
            sample_rate: Target sample rate (will resample if different)
            mean: Normalization mean
            std: Normalization std
            num_clips: Number of clips to extract per sample for multi-clip eval
            clip_overlap: Allow overlapping clips when extracting multiple
            training: Whether in training mode (affects clip sampling)
            debug: Return additional debug info
        """
        self.audio_data_dir = audio_data_dir
        self.dataset_csv = dataset_csv
        self.vocabulary_csv = vocabulary_csv

        # Audio processing params
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.mean = mean
        self.std = std

        # Multi-clip params
        self.num_clips = num_clips
        self.clip_overlap = clip_overlap
        self.training = training
        self.debug = debug

        # Process dataset
        self.process_dataset()

    def process_dataset(self):
        """Load and process dataset CSV and vocabulary."""
        # Load vocabulary (index -> mid mapping)
        # vocabulary.csv format: index,label_name,mid
        mid_to_index = {}
        index_to_name = {}
        with open(self.vocabulary_csv, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    try:
                        idx = int(row[0])
                        name = row[1]
                        mid = row[2]
                        mid_to_index[mid] = idx
                        index_to_name[idx] = name
                    except ValueError:
                        continue

        self.num_classes = len(mid_to_index)
        self.mid_to_index = mid_to_index
        self.index_to_name = index_to_name
        logger.info(f"Loaded {self.num_classes} classes from vocabulary")

        # Load dataset CSV
        # dev.csv format: fname,labels,mids,split
        # eval.csv format: fname,labels,mids
        existing_files = set(os.listdir(self.audio_data_dir))

        self.dataset = []
        skipped = 0

        with open(self.dataset_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["fname"]
                mids = row["mids"].split(",")

                # Check if audio file exists
                audio_filename = f"{fname}.wav" if not fname.endswith(".wav") else fname
                if audio_filename not in existing_files:
                    skipped += 1
                    continue

                audio_path = os.path.join(self.audio_data_dir, audio_filename)

                # Create multi-hot label vector
                multi_hot = torch.zeros(self.num_classes)
                for mid in mids:
                    mid = mid.strip()
                    if mid in mid_to_index:
                        multi_hot[mid_to_index[mid]] = 1.0

                self.dataset.append((audio_path, multi_hot))

        logger.info(f"Loaded {len(self.dataset)} samples, skipped {skipped} (missing files)")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        audio_path, multi_hot_y = self.dataset[index]

        # Try to load audio, resample if needed
        audio_data = None
        attempts = 0
        max_attempts = 10

        while audio_data is None and attempts < max_attempts:
            try:
                audio_data = self.load_and_transform_audio(audio_path)
            except Exception as e:
                logger.warning(f"Error loading {audio_path}: {e}")
                # Try a different sample
                index = np.random.randint(len(self.dataset))
                audio_path, multi_hot_y = self.dataset[index]
                attempts += 1

        if audio_data is None:
            # Return zeros if all attempts failed
            audio_data = torch.zeros(self.num_clips, 1, self.target_length, self.num_mel_bins)

        if self.debug:
            label_indices = multi_hot_y.nonzero().flatten().tolist()
            label_names = [self.index_to_name.get(i, "unknown") for i in label_indices]
            return {"audio": audio_data}, multi_hot_y, audio_path, label_names
        else:
            return {"audio": audio_data}, multi_hot_y

    def load_and_transform_audio(self, audio_path):
        """
        Load audio and convert to mel spectrogram clips.

        For multi-clip evaluation:
        - If audio is shorter than target_length, pad with zeros
        - If audio is longer, extract num_clips clips (evenly spaced or random)

        Returns:
            Tensor of shape (num_clips, 1, target_length, num_mel_bins)
        """
        # Load audio
        wav, sr = torchaudio.load(audio_path)

        # Resample to target sample rate if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Normalize waveform
        wav = wav - wav.mean()

        # Compute mel spectrogram
        mel = torchaudio.compliance.kaldi.fbank(
            wav,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_shift=10,
        )  # Shape: (T, num_mel_bins)

        n_frames = mel.shape[0]

        # Extract clips
        clips = []
        if n_frames <= self.target_length:
            # Pad with zeros if shorter
            padded = torch.zeros(self.target_length, self.num_mel_bins)
            padded[:n_frames, :] = mel
            mel_clip = padded.unsqueeze(0)  # (1, T, F)
            # Duplicate for all clips
            for _ in range(self.num_clips):
                clips.append(mel_clip.clone())
        else:
            # Extract multiple clips
            if self.num_clips == 1:
                # Single clip: take center crop
                start = (n_frames - self.target_length) // 2
                mel_clip = mel[start : start + self.target_length, :].unsqueeze(0)
                clips.append(mel_clip)
            else:
                if self.training:
                    # Random clips during training
                    for _ in range(self.num_clips):
                        max_start = n_frames - self.target_length
                        start = np.random.randint(0, max_start + 1)
                        mel_clip = mel[start : start + self.target_length, :].unsqueeze(0)
                        clips.append(mel_clip)
                else:
                    # Evenly spaced clips during evaluation
                    if self.clip_overlap:
                        # Allow overlap for more coverage
                        stride = (n_frames - self.target_length) / max(1, self.num_clips - 1)
                        for i in range(self.num_clips):
                            start = int(i * stride)
                            start = min(start, n_frames - self.target_length)
                            mel_clip = mel[start : start + self.target_length, :].unsqueeze(0)
                            clips.append(mel_clip)
                    else:
                        # Non-overlapping clips
                        stride = n_frames // self.num_clips
                        for i in range(self.num_clips):
                            start = i * stride
                            if start + self.target_length > n_frames:
                                start = n_frames - self.target_length
                            mel_clip = mel[start : start + self.target_length, :].unsqueeze(0)
                            clips.append(mel_clip)

        # Stack clips: (num_clips, 1, T, F)
        clips = torch.stack(clips, dim=0)

        # Normalize
        clips = (clips - self.mean) / (self.std * 2)

        return clips


class FSD50KMultiClipCollator:
    """
    Custom collator for multi-clip FSD50K dataset.

    Handles batching of multi-clip audio data where each sample
    has shape (num_clips, 1, T, F).
    """

    def __init__(self, flatten_clips=False):
        """
        Args:
            flatten_clips: If True, flatten clips into batch dimension
                          (B, num_clips, 1, T, F) -> (B*num_clips, 1, T, F)
        """
        self.flatten_clips = flatten_clips

    def __call__(self, batch):
        """
        Collate batch of samples.

        Args:
            batch: List of (audio_dict, labels) or (audio_dict, labels, path, names)

        Returns:
            Collated batch with audio shape (B, num_clips, 1, T, F) or
            (B*num_clips, 1, T, F) if flatten_clips=True
        """
        audio_data = torch.stack([item[0]["audio"] for item in batch], dim=0)
        labels = torch.stack([item[1] for item in batch], dim=0)

        if self.flatten_clips:
            B, num_clips, C, T, F = audio_data.shape
            audio_data = audio_data.view(B * num_clips, C, T, F)

        result = ({"audio": audio_data}, labels)

        # Handle debug info if present
        if len(batch[0]) > 2:
            paths = [item[2] for item in batch]
            names = [item[3] for item in batch]
            result = result + (paths, names)

        return result


def make_fsd50k_dataloader(
    audio_dir,
    csv_path,
    vocabulary_csv,
    batch_size,
    world_size,
    rank,
    num_mel_bins=128,
    target_length=512,
    sample_rate=16000,
    audio_mean=-4.268,
    audio_std=4.569,
    num_clips=1,
    training=False,
    num_workers=12,
):
    """
    Create dataloader for FSD50K dataset.

    Args:
        audio_dir: Directory with audio files
        csv_path: Path to dataset CSV (dev.csv or eval.csv)
        vocabulary_csv: Path to vocabulary.csv
        batch_size: Batch size per GPU
        world_size: Number of distributed processes
        rank: Current process rank
        num_mel_bins: Number of mel bins
        target_length: Target length in mel frames
        sample_rate: Target sample rate
        audio_mean: Normalization mean
        audio_std: Normalization std
        num_clips: Number of clips per sample
        training: Whether in training mode
        num_workers: Number of dataloader workers

    Returns:
        (dataloader, sampler)
    """
    dataset = FSD50KEval(
        audio_data_dir=audio_dir,
        dataset_csv=csv_path,
        vocabulary_csv=vocabulary_csv,
        num_mel_bins=num_mel_bins,
        target_length=target_length,
        sample_rate=sample_rate,
        mean=audio_mean,
        std=audio_std,
        num_clips=num_clips,
        training=training,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
    )

    # Use custom collator for multi-clip
    collator = FSD50KMultiClipCollator(flatten_clips=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=training,
        collate_fn=collator,
    )

    return data_loader, sampler
