# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Common utilities for audio classification evaluation.

Shared components:
- Augmentations: mixup, specaugment, time shift
- EMA: Exponential Moving Average for model weights
- Schedulers: WarmupCosine LR, Cosine weight decay
- Utilities: gather_tensor, compute_mAP, load_checkpoint
"""

import logging
import math

import numpy as np
import torch

from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger(__name__)


# =============================================================================
# AUGMENTATION UTILITIES
# =============================================================================


def mixup_data(x, y, alpha=0.4):
    """
    Mixup augmentation for audio data and labels.

    Args:
        x: Input tensor of shape (B, 1, T, F) - batch of mel spectrograms
        y: Target tensor - multi-hot labels (B, num_classes) or one-hot for single-label
        alpha: Mixup interpolation coefficient (Beta distribution parameter)

    Returns:
        mixed_x: Mixed input
        mixed_y: Mixed labels (soft labels)
        lam: Lambda value used for mixing
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]

    return mixed_x, mixed_y, lam


def spec_augment(mel, time_mask_param=50, freq_mask_param=10, num_time_masks=2, num_freq_masks=2):
    """
    SpecAugment: Apply time and frequency masking to mel spectrogram.

    Args:
        mel: Input tensor of shape (B, 1, T, F) - batch of mel spectrograms
        time_mask_param: Maximum width of time mask
        freq_mask_param: Maximum width of frequency mask
        num_time_masks: Number of time masks to apply
        num_freq_masks: Number of frequency masks to apply

    Returns:
        Augmented mel spectrogram
    """
    mel = mel.clone()
    batch_size, channels, time_steps, freq_bins = mel.shape

    for i in range(batch_size):
        # Time masking
        for _ in range(num_time_masks):
            t = np.random.randint(0, min(time_mask_param, time_steps))
            t0 = np.random.randint(0, max(1, time_steps - t))
            mel[i, :, t0 : t0 + t, :] = 0

        # Frequency masking
        for _ in range(num_freq_masks):
            f = np.random.randint(0, min(freq_mask_param, freq_bins))
            f0 = np.random.randint(0, max(1, freq_bins - f))
            mel[i, :, :, f0 : f0 + f] = 0

    return mel


def random_time_shift(mel, max_shift_ratio=0.2):
    """
    Random time shifting (rolling) of mel spectrogram.

    Args:
        mel: Input tensor of shape (B, 1, T, F) - batch of mel spectrograms
        max_shift_ratio: Maximum shift as a ratio of time dimension

    Returns:
        Time-shifted mel spectrogram
    """
    batch_size, channels, time_steps, freq_bins = mel.shape
    max_shift = int(time_steps * max_shift_ratio)

    shifted_mel = mel.clone()
    for i in range(batch_size):
        shift = np.random.randint(-max_shift, max_shift + 1)
        shifted_mel[i] = torch.roll(mel[i], shifts=shift, dims=1)

    return shifted_mel


def volume_jitter(mel, jitter_range=0.2):
    """
    Random volume scaling (gain augmentation) for mel spectrograms.

    Applies a random multiplicative scale factor to each sample in the batch.
    This helps the model become invariant to amplitude variations.

    Commonly used in AST, BEATs, and PaSST papers.

    Args:
        mel: Input tensor of shape (B, 1, T, F) - batch of mel spectrograms
        jitter_range: Maximum deviation from 1.0 (e.g., 0.2 means scale in [0.8, 1.2])

    Returns:
        Volume-jittered mel spectrogram
    """
    batch_size = mel.shape[0]
    # Generate random scale factors in [1 - jitter_range, 1 + jitter_range]
    scale = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=mel.device) * 2 - 1) * jitter_range
    return mel * scale


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that is updated as:
        shadow = decay * shadow + (1 - decay) * current_params
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """Update shadow parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """Restore original parameters after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]


# =============================================================================
# LEARNING RATE AND WEIGHT DECAY SCHEDULERS
# =============================================================================


class WarmupCosineLRSchedule:
    """Warmup + Cosine annealing learning rate schedule."""

    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule:
    """Cosine weight decay schedule."""

    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd


# =============================================================================
# OPTIMIZER INITIALIZATION
# =============================================================================


def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    """Initialize optimizers and schedulers for all classifier heads."""
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []

    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup", 0) * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        scalers += [torch.cuda.amp.GradScaler() if use_bfloat16 else None]

    return optimizers, scalers, schedulers, wd_schedulers


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================


def load_checkpoint(r_path, classifiers, opt, scaler, val_only=False):
    """Load training checkpoint."""
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    # -- loading classifiers
    pretrained_dict = checkpoint["classifiers"]
    msg = []
    for c, pd in zip(classifiers, pretrained_dict):
        # Handle module. prefix mismatch (DDP vs non-DDP)
        model_state = c.state_dict()

        # Check if we need to add or remove module. prefix
        model_has_module = any(k.startswith("module.") for k in model_state.keys())
        ckpt_has_module = any(k.startswith("module.") for k in pd.keys())

        if model_has_module and not ckpt_has_module:
            # Model has module. prefix but checkpoint doesn't - add it
            pd = {"module." + k: v for k, v in pd.items()}
        elif not model_has_module and ckpt_has_module:
            # Model doesn't have module. prefix but checkpoint does - remove it
            pd = {k.replace("module.", ""): v for k, v in pd.items()}

        msg.append(c.load_state_dict(pd))

    if val_only:
        logger.info(f"loaded pretrained classifier with msg: {msg}")
        return classifiers, opt, scaler, 0, checkpoint.get("best_mAP", 0.0)

    epoch = checkpoint["epoch"]
    best_mAP = checkpoint.get("best_mAP", 0.0)
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    if scaler is not None and checkpoint["scaler"] is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"])]

    logger.info(f"loaded optimizers from epoch {epoch}")

    return classifiers, opt, scaler, epoch, best_mAP


# =============================================================================
# DISTRIBUTED UTILITIES
# =============================================================================


def gather_tensor(tensor):
    """Gather tensor from all processes."""
    try:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    except Exception:
        return tensor

    if world_size == 1:
        return tensor

    tensor = tensor.cuda()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor)
    gathered = torch.cat(gathered, dim=0)

    if rank == 0:
        return gathered.cpu()
    return None


# =============================================================================
# LOSS FUNCTIONS FOR CLASS IMBALANCE
# =============================================================================


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal loss down-weights well-classified examples and focuses on hard,
    misclassified examples. This is particularly useful for imbalanced datasets
    like SUPERB keyword spotting where "unknown" class dominates.

    Loss = -alpha * (1 - pt)^gamma * log(pt)

    where pt = p if y=1 else (1-p), and p is the model's predicted probability.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)

    Args:
        alpha: Weighting factor for the rare class (default: 1.0)
               Can also be a tensor of per-class weights
        gamma: Focusing parameter (default: 2.0)
               gamma=0 is equivalent to cross-entropy
               gamma>0 reduces the relative loss for well-classified examples
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Label smoothing factor (default: 0.0)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C) where C is number of classes
            targets: Ground truth class indices of shape (N,)

        Returns:
            Focal loss value
        """
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)  # pt = probability of correct class

        # Apply focal modulation
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedFocalLoss(torch.nn.Module):
    """
    Class-Balanced Focal Loss combining per-class weights with focal loss.

    This combines two strategies:
    1. Per-class weights (inverse frequency) to handle class imbalance
    2. Focal loss to focus on hard examples

    Effective for highly imbalanced datasets like SUPERB keyword spotting.

    Args:
        class_weights: Tensor of per-class weights (shape: num_classes)
        gamma: Focal loss focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Label smoothing factor (default: 0.0)
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Ground truth class indices of shape (N,)

        Returns:
            Class-balanced focal loss value
        """
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)

        # Get per-sample class weights
        sample_weights = self.class_weights[targets]

        # Apply focal modulation with class weights
        focal_weight = sample_weights * ((1 - pt) ** self.gamma)
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(
    class_counts: torch.Tensor,
    mode: str = "inverse",
    beta: float = 0.9999,
) -> torch.Tensor:
    """
    Compute class weights for handling class imbalance.

    Args:
        class_counts: Tensor of sample counts per class (shape: num_classes)
        mode: Weighting strategy
            - "inverse": weights = 1 / count (normalized)
            - "inverse_sqrt": weights = 1 / sqrt(count) (normalized)
            - "effective": Class-Balanced Loss (Cui et al., 2019)
                          weights = (1 - beta) / (1 - beta^n)
        beta: Beta parameter for "effective" mode (default: 0.9999)

    Returns:
        Normalized class weights tensor
    """
    class_counts = class_counts.float()
    num_classes = len(class_counts)

    if mode == "inverse":
        # Inverse frequency weighting
        weights = 1.0 / (class_counts + 1e-6)
    elif mode == "inverse_sqrt":
        # Square root inverse frequency (gentler)
        weights = 1.0 / (torch.sqrt(class_counts) + 1e-6)
    elif mode == "effective":
        # Class-Balanced Loss: (1 - beta) / (1 - beta^n)
        # Reference: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples"
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'inverse', 'inverse_sqrt', 'effective'")

    # Normalize so weights sum to num_classes (keeps loss scale similar)
    weights = weights / weights.sum() * num_classes

    return weights


def create_weighted_sampler(
    dataset,
    class_counts: torch.Tensor = None,
) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for class-balanced sampling.

    This sampler ensures each class is equally represented per epoch by
    sampling minority classes more frequently, WITHOUT duplicating data.

    Args:
        dataset: Dataset with .samples attribute containing (path, label) tuples
        class_counts: Optional pre-computed class counts. If None, computed from dataset.

    Returns:
        WeightedRandomSampler instance
    """
    # Get labels from dataset
    if hasattr(dataset, "samples"):
        labels = [label for _, label in dataset.samples]
    else:
        raise ValueError("Dataset must have .samples attribute with (path, label) tuples")

    labels = torch.tensor(labels)
    num_classes = labels.max().item() + 1

    # Compute class counts if not provided
    if class_counts is None:
        class_counts = torch.zeros(num_classes)
        for label in labels:
            class_counts[label] += 1

    # Compute per-sample weights (inverse of class frequency)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]

    # Create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(dataset),
        replacement=True,  # Required for weighted sampling
    )

    return sampler


def get_class_counts_from_dataset(dataset) -> torch.Tensor:
    """
    Count samples per class in a dataset.

    Args:
        dataset: Dataset with .samples attribute or indexable (data, label) pairs

    Returns:
        Tensor of class counts
    """
    if hasattr(dataset, "samples"):
        labels = [label for _, label in dataset.samples]
    else:
        # Fallback: iterate through dataset
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)

    labels = torch.tensor(labels)
    num_classes = labels.max().item() + 1

    class_counts = torch.zeros(num_classes)
    for label in labels:
        class_counts[label] += 1

    return class_counts


# =============================================================================
# METRICS
# =============================================================================


def compute_mAP(predictions, targets):
    """Compute mean Average Precision and mean AUC for multi-label classification."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    # predictions: (N, num_classes)
    # targets: (N, num_classes) multi-hot vectors

    APs = []
    AUCs = []

    num_classes = predictions.shape[1]
    for i in range(num_classes):
        if targets[:, i].sum() > 0:  # Only compute for classes with positive samples
            try:
                ap = average_precision_score(targets[:, i], predictions[:, i])
                APs.append(ap)
            except Exception:
                pass
            try:
                auc = roc_auc_score(targets[:, i], predictions[:, i])
                AUCs.append(auc)
            except Exception:
                pass

    mAP = np.mean(APs) if APs else 0.0
    mAUC = np.mean(AUCs) if AUCs else 0.0

    return mAP, mAUC
