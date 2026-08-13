# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import os
from logging import getLogger

import numpy as np
import torch
import torchaudio

_GLOBAL_SEED = 0
logger = getLogger()


class AudioSetAudioEval(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_data_dir,
        dataset_csv,
        num_mel_bins=128,  # fixed as per standards
        target_length=1024,  # standard for 10s audio (10 frames/s), rounded to a power of 2
        sample_rate=16000,  # fixed as per standards
        mean=-4.268,  # # fixed as per standards
        std=4.569,  # this is 1xstd because in audio dataset, we divide by 2xstd # fixed as per standards
        debug=False,
        audio_only=True,
        debug_only=False,
    ):

        assert audio_only is True
        self.audio_only = audio_only
        self.debug_only = debug_only  # do not load data, just return paths and labels
        # data
        self.audio_data_dir = audio_data_dir
        self.dataset_csv = dataset_csv

        self.process_dataset()

        # to analyze
        self.debug = debug

        # transforming audio data
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.mean = mean
        self.std = std

    def process_dataset(self):
        # loading dataset
        file_path_labels = []
        label_set = []

        ontology_path = os.path.join(os.path.dirname(self.dataset_csv), "ontology.json")
        if os.path.exists(ontology_path):
            with open(ontology_path, "r") as fp:
                ontology = json.load(fp)
        else:
            ontology = []
            logger.warning(f"ontology.json not found at {ontology_path}, class names will not be available")

        label_to_class = {entry["id"]: entry["name"] for entry in ontology}

        existing_audio_files = set(os.listdir(self.audio_data_dir))

        with open(self.dataset_csv, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                # Skip commented lines
                if row[0][0] == "#":
                    continue

                yt_id, start_time, end_time = row[:3]
                labels = [label.strip().strip('"') for label in row[3:]]

                label_set.extend(labels)

                start_time_ms = int(float(start_time) * 1000)
                end_time_ms = int(float(end_time) * 1000)

                audio_file_name = f"{yt_id}_{start_time_ms}_{end_time_ms}.flac"

                audio_file_path = os.path.join(self.audio_data_dir, audio_file_name)

                if audio_file_name in existing_audio_files:
                    file_path_labels.append((audio_file_path, labels))

        label_set = sorted(list(set(label_set)))  # sorting to make sure both eval and train have same
        self.num_classes = len(label_set)
        label_to_index = {label: i for i, label in enumerate(label_set)}
        self.index_to_label = {i: label for label, i in label_to_index.items()}
        self.index_to_class_name = {i: label_to_class.get(label, label) for i, label in self.index_to_label.items()}

        self.dataset = []
        for audio_file_path, labels in file_path_labels:
            multi_hot_vector = torch.zeros(len(label_set))
            indices = torch.tensor([label_to_index[label] for label in labels])
            multi_hot_vector.scatter_(0, indices, 1)

            self.dataset.append((audio_file_path, multi_hot_vector))

    def load_and_transform_audio_data(
        self,
        audio_path,
    ):

        # using standard audio processing used in all papers on audioset
        # NOTE: need to change when we have shorter audio
        try:
            mel = preprocess_audio(audio_path, self.target_length, self.mean, self.std)
        except Exception:
            return None

        return mel

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        audio_file_path, multi_hot_y = self.dataset[index]

        audio_data = None
        loaded_sample = False

        while not loaded_sample:

            if self.debug_only is True:
                # do not load any audio/video data and return only paths and labels
                break

            audio_data = self.load_and_transform_audio_data(audio_file_path)
            if audio_data is None:  # trigger resample
                index = np.random.randint(self.__len__())
                audio_file_path, multi_hot_y = self.dataset[index]
                continue

            if audio_data is not None:
                loaded_sample = True

        # NOTE: some of the files may give error / or large gap b/w audio frames and target length warning.
        if self.debug:
            label_indices = multi_hot_y.nonzero().flatten().tolist()
            label_classes = [self.index_to_class_name[i] for i in label_indices]

            return {"audio": audio_data}, multi_hot_y, audio_file_path, label_classes
        else:
            return {"audio": audio_data}, multi_hot_y


#################################
# helper functions
#################################
# ===== Audio Preprocessing =====
def preprocess_audio(path, target_length, norm_mean, norm_std):
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
    if n_frames < target_length:
        mel = torch.nn.ZeroPad2d((0, 0, 0, target_length - n_frames))(mel)
    elif n_frames > target_length:
        mel = mel[:, :target_length, :]

    mel = (mel - norm_mean) / (norm_std * 2)
    return mel  # shape [1, T, F]


def _tensor_normalize_inplace(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize (with dimensions C, T, H, W).
        mean (tensor): mean value to subtract (in 0 to 255 floats).
        std (tensor): std to divide (in 0 to 255 floats).
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()

    C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)  # Make C the last dimension
    tensor.sub_(mean).div_(std)
    tensor = tensor.permute(1, 0).view(C, T, H, W)  # Put C back in front
    return tensor
