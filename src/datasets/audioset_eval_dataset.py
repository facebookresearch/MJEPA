# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import math
import os
import warnings
from logging import getLogger

import numpy as np
import torch
import torchaudio
import torchvision
from decord import VideoReader, cpu

from src.datasets.utils.dataloader import ConcatIndices

_GLOBAL_SEED = 0
logger = getLogger()


class AudioSetEval(torch.utils.data.Dataset):
    """AudioSet Eval dataset."""

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        # video params
        frames_per_clip=16,
        fps=4,
        dataset_fpcs=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
        # audio params
        num_mel_bins=128,  # fixed as per standards
        target_length=1024,  # standard for 10s audio (10 frames/s), rounded to a power of 2
        audio_sample_rate=16000,  # fixed as per standards
        audio_mean=-4.268,  # # fixed as per standards
        audio_std=4.569,  # 1x std; audio normalization divides by 2x std
    ):
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps

        # transforming audio data
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.audio_sample_rate = audio_sample_rate
        self.audio_mean = audio_mean
        self.audio_std = audio_std

        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}.")

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError("Frames per clip not properly specified for NFS data paths")
            self.dataset_fpcs = dataset_fpcs

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths and labels
        ontology_path = os.path.join(os.path.dirname(self.data_paths[0]), "ontology.json")
        if os.path.exists(ontology_path):
            with open(ontology_path, "r") as fp:
                ontology = json.load(fp)
        else:
            ontology = []
            logger.warning(f"ontology.json not found at {ontology_path}, class names will not be available")

        label_to_class = {entry["id"]: entry["name"] for entry in ontology}

        audio_samples, video_samples, label_samples = [], [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:
            print("Data path: ", data_path)
            assert data_path[-4:] == ".csv"
            labels_set = []
            file_path_labels = []

            with open(data_path, "r") as f:
                reader = csv.reader(f, delimiter=" ")
                for row in reader:
                    audio_path = row[0]
                    video_path = row[1]

                    labels = [label.strip() for label in row[2:]]
                    labels_set.extend(labels)

                    file_path_labels.append((audio_path, video_path, labels))

            labels_set = sorted(list(set(labels_set)))  # sorting to make sure both eval and train have same
            self.num_classes = len(labels_set)
            label_to_index = {label: i for i, label in enumerate(labels_set)}
            self.index_to_label = {i: label for label, i in label_to_index.items()}
            self.index_to_class_name = {
                i: label_to_class.get(label, label) for i, label in self.index_to_label.items()
            }

            for audio_file_path, video_file_path, labels in file_path_labels:
                multi_hot_vector = torch.zeros(len(labels_set))
                indices = torch.tensor([label_to_index[label] for label in labels])
                multi_hot_vector.scatter_(0, indices, 1)

                audio_samples.append(audio_file_path)
                video_samples.append(video_file_path)
                label_samples.append(multi_hot_vector)

            num_samples = len(file_path_labels)
            self.num_samples_per_dataset.append(num_samples)

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.audio_samples = audio_samples
        self.video_samples = video_samples
        self.label_samples = label_samples

        assert len(self.audio_samples) == len(self.video_samples) == len(self.label_samples)

    def __getitem__(self, index):
        loaded_sample = False
        audio_data = None
        video_data = None
        label = None

        # Keep trying to load audio/videos until you find a valid sample
        while not loaded_sample:
            label = self.label_samples[index]

            audio_data = self.get_item_audio(index)
            if audio_data is None:  # trigger resample
                index = np.random.randint(self.__len__())
                continue

            video_data = self.get_item_video(index)
            if video_data is None:  # trigger resample
                index = np.random.randint(self.__len__())
                continue

            if audio_data is not None and video_data is not None:
                loaded_sample = True
            else:
                loaded_sample = False

        return {
            "audio": audio_data,
            "video": video_data,
        }, label

    def get_item_audio(self, index):
        audio_path = self.audio_samples[index]

        try:
            mel = self.preprocess_audio(audio_path)
        except Exception:
            return None

        return mel

    def preprocess_audio(self, path):
        wav, sr = torchaudio.load(path)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav - wav.mean()

        mel = torchaudio.compliance.kaldi.fbank(
            wav,  # wav.unsqueeze(0) = need a batch dim if using soundfile
            htk_compat=True,
            sample_frequency=16000,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        ).unsqueeze(0)

        # pad or truncate
        n_frames = mel.shape[1]
        if n_frames < self.target_length:
            mel = torch.nn.ZeroPad2d((0, 0, 0, self.target_length - n_frames))(mel)
        elif n_frames > self.target_length:
            mel = mel[:, : self.target_length, :]

        mel = (mel - self.audio_mean) / (self.audio_std * 2)
        return mel  # shape [1, T, F]

    def get_item_video(self, index):
        sample = self.video_samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_idx]

        buffer, clip_indices = self.loadvideo_decord(sample, frames_per_clip)  # [T H W 3]
        loaded_video = len(buffer) > 0
        if not loaded_video:
            return

        def split_into_clips(video):
            """Split video into a list of clips"""
            fpc = frames_per_clip
            nc = self.num_clips
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, clip_indices

    def get_item_image(self, index):
        sample = self.video_samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]

        try:
            image_tensor = torchvision.io.read_image(path=sample, mode=torchvision.io.ImageReadMode.RGB)
        except Exception:
            return
        clip_indices = [np.arange(start=0, stop=fpc, dtype=np.int32)]

        # Expanding the input image [3, H, W] ==> [T, 3, H, W]
        buffer = image_tensor.unsqueeze(dim=0).repeat((fpc, 1, 1, 1))
        buffer = buffer.permute((0, 2, 3, 1))  # [T, 3, H, W] ==> [T H W 3]

        if self.shared_transform is not None:
            # Technically we can have only transform, doing this just for the sake of consistency with videos.
            buffer = self.shared_transform(buffer)

        if self.transform is not None:
            buffer = [self.transform(buffer)]

        return buffer, clip_indices

    def loadvideo_decord(self, sample, fpc):
        """Load video content using Decord"""

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f"video path not found {fname=}")
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize > self.filter_long_videos:
            warnings.warn(f"skipping long video of size {_fsize=} (bytes)")
            return [], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fstp = self.frame_step
        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)

            if self.duration is not None:
                assert self.fps is None
                fstp = int(self.duration * video_fps / fpc)
            else:
                assert self.duration is None
                fstp = video_fps // self.fps

        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"skipping video of length {len(vr)}")
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - partition_len // fstp) * partition_len,
                        )
                    )
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - sample_len // fstp) * sample_len,
                        )
                    )
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.audio_samples)
