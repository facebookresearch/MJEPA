# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math


class WSDSchedule(object):

    def __init__(self, optimizer, warmup_steps, anneal_steps, T_max, start_lr, ref_lr, final_lr=0.0):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.anneal_steps = anneal_steps
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps - anneal_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        elif self._step < self.T_max + self.warmup_steps:
            new_lr = self.ref_lr
        else:
            _step = self._step - (self.T_max + self.warmup_steps)
            progress = float(_step) / float(max(1, self.anneal_steps))
            new_lr = self.ref_lr + progress * (self.final_lr - self.ref_lr)

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
            if "lr_scale" in group:
                group["lr"] *= group["lr_scale"]

        return new_lr


class WarmupCosineSchedule(object):

    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, last_epoch=-1, final_lr=0.0):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(
                self.final_lr,
                self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

        return new_lr


class AVWarmupCosineSchedule(object):

    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, last_epoch=-1, final_lr=0.0):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.0

        # video with a short schedule | stops training early
        self.video_T_max = self.T_max // 2
        self.video_final_lr = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
            video_new_lr = new_lr
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(
                self.final_lr,
                self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )

            # -- video progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.video_T_max))
            video_new_lr = max(
                self.video_final_lr,
                self.video_final_lr + (self.ref_lr - self.video_final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        for group in self.optimizer.param_groups:
            if group["mod_type"] == "audio":
                group["lr"] = new_lr
            elif group["mod_type"] == "video":
                group["lr"] = video_new_lr
            else:
                assert False, "mod type not defined"

        return new_lr, video_new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        return new_wd


class LinearDecaySchedule(object):
    """Linear decay schedule for cooldown/annealing phase.

    Linearly interpolates from ref_lr to final_lr over T_max steps.
    No warmup - starts directly at ref_lr.
    """

    def __init__(self, optimizer, ref_lr, T_max, final_lr=0.0):
        self.optimizer = optimizer
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.T_max = T_max
        self._step = 0

    def step(self):
        self._step += 1
        progress = float(self._step) / float(max(1, self.T_max))
        new_lr = self.ref_lr + progress * (self.final_lr - self.ref_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

        return new_lr


class AVCosineWDSchedule(object):

    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.0

        # video with a short schedule | stops training early
        self.video_T_max = self.T_max // 2

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        video_progress = self._step / self.video_T_max

        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
        video_new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * video_progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
            video_new_wd = max(self.final_wd, video_new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)
            video_new_wd = min(self.final_wd, video_new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                if group["mod_type"] == "audio":
                    group["weight_decay"] = new_wd
                elif group["mod_type"] == "video":
                    group["weight_decay"] = video_new_wd
                else:
                    assert False, "mod type not defined"
        return new_wd, video_new_wd
