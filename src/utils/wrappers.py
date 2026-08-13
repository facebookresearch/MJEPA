# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class MultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None):
        """
        :param x: [list] List of Tensors of different seq lengths
        :param masks: [list[list]] List of Tensors
            (out index: masks for given seq length,
            inner index: multimasks for that seq len)
        """
        if masks is None:
            return [self.backbone(xi) for xi in x]

        outs = [[] for _ in x]
        for i, (xi, mi) in enumerate(zip(x, masks)):
            for mij in mi:
                outs[i] += [self.backbone(xi, masks=mij)]
        return outs


class MJEPAMultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        audio_x=None,
        audio_masks=None,
        video_x=None,
        video_masks=None,
        training_mode=False,
    ):
        """
        :param x: [list] List of Tensors of different seq lengths
        :param masks: [list[list]] List of Tensors
            (out index: masks for given seq length,
            inner index: multimasks for that seq len)
        ^ similar for both audio and video
        NOTE: not considering the case where one mask is present but the other is not.
        """
        if video_x is None and audio_x is None:
            assert False, "No input to the encoder. Both video and audio are none"

        audio_outs = None
        video_outs = None

        if audio_x is not None and video_x is not None:
            audio_outs, video_outs = [], []

            if audio_masks is None and video_masks is None:
                for audio_xi, video_xi in zip(audio_x, video_x):
                    audio_out, video_out = self.backbone(audio_x=audio_xi, video_x=video_xi, training=training_mode)
                    audio_outs.append(audio_out)
                    video_outs.append(video_out)
                return audio_outs, video_outs

            elif audio_masks is not None and video_masks is not None:
                audio_outs = [[] for _ in audio_x]
                video_outs = [[] for _ in video_x]

                for i, (audio_xi, audio_mi, video_xi, video_mi) in enumerate(
                    zip(audio_x, audio_masks, video_x, video_masks)
                ):
                    for audio_mij, video_mij in zip(audio_mi, video_mi):
                        audio_out, video_out = self.backbone(
                            audio_x=audio_xi,
                            video_x=video_xi,
                            audio_masks=audio_mij,
                            video_masks=video_mij,
                            training=training_mode,
                        )
                        audio_outs[i].append(audio_out)
                        video_outs[i].append(video_out)
            else:
                assert False, "either both masks are there or no mask is there"

        elif audio_x is not None and video_x is None:
            audio_outs = []

            if audio_masks is None:
                for audio_xi in audio_x:
                    audio_out, _ = self.backbone(audio_x=audio_xi, training=training_mode)
                    audio_outs.append(audio_out)
                return audio_outs, None

            else:
                audio_outs = [[] for _ in audio_x]

                for i, (audio_xi, audio_mi) in enumerate(zip(audio_x, audio_masks)):
                    for audio_mij in audio_mi:
                        audio_out, _ = self.backbone(audio_x=audio_xi, audio_masks=audio_mij, training=training_mode)
                        audio_outs[i].append(audio_out)

        elif audio_x is None and video_x is not None:
            video_outs = []

            if video_masks is None:
                for video_xi in video_x:
                    _, video_out = self.backbone(video_x=video_xi, training=training_mode)
                    video_outs.append(video_out)
                return None, video_outs

            else:
                video_outs = [[] for _ in video_x]

                for i, (video_xi, video_mi) in enumerate(zip(video_x, video_masks)):
                    for video_mij in video_mi:
                        _, video_out = self.backbone(video_x=video_xi, video_masks=video_mij, training=training_mode)
                        video_outs[i].append(video_out)

        return audio_outs, video_outs


class PredictorMultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y, has_cls=False):
        n = 0
        outs = [[] for _ in x]
        for i, (xi, mxi, myi) in enumerate(zip(x, masks_x, masks_y)):
            for xij, mxij, myij in zip(xi, mxi, myi):
                outs[i] += [self.backbone(xij, mxij, myij, mask_index=i, has_cls=has_cls)]
                n += 1
        return outs


class MJEPAPredictorMultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        audio_x=None,
        audio_masks_x=None,
        audio_masks_y=None,
        video_x=None,
        video_masks_x=None,
        video_masks_y=None,
        mask_index=1,
    ):

        audio_outs = None
        video_outs = None

        if video_x is None and audio_x is None:
            assert False, "No input to the encoder. Both video and audio are none"

        if audio_x is not None and video_x is not None:
            audio_outs = [[] for _ in audio_x]
            video_outs = [[] for _ in video_x]

            for i, (audio_xi, audio_mxi, audio_myi, video_xi, video_mxi, video_myi) in enumerate(
                zip(audio_x, audio_masks_x, audio_masks_y, video_x, video_masks_x, video_masks_y)
            ):
                for audio_xij, audio_mxij, audio_myij, video_xij, video_mxij, video_myij in zip(
                    audio_xi, audio_mxi, audio_myi, video_xi, video_mxi, video_myi
                ):
                    audio_out, video_out = self.backbone(
                        audio_x=audio_xij,
                        audio_masks_x=audio_mxij,
                        audio_masks_y=audio_myij,
                        video_x=video_xij,
                        video_masks_x=video_mxij,
                        video_masks_y=video_myij,
                        mask_index=i,
                    )
                    audio_outs[i].append(audio_out)
                    video_outs[i].append(video_out)

        elif audio_x is not None and video_x is None:
            audio_outs = [[] for _ in audio_x]
            for i, (audio_xi, audio_mxi, audio_myi) in enumerate(zip(audio_x, audio_masks_x, audio_masks_y)):
                for audio_xij, audio_mxij, audio_myij in zip(audio_xi, audio_mxi, audio_myi):
                    audio_out, _ = self.backbone(
                        audio_x=audio_xij, audio_masks_x=audio_mxij, audio_masks_y=audio_myij, mask_index=i
                    )
                    audio_outs[i].append(audio_out)

        elif audio_x is None and video_x is not None:
            video_outs = [[] for _ in video_x]
            for i, (video_xi, video_mxi, video_myi) in enumerate(zip(video_x, video_masks_x, video_masks_y)):
                for video_xij, video_mxij, video_myij in zip(video_xi, video_mxi, video_myi):
                    _, video_out = self.backbone(
                        video_x=video_xij, video_masks_x=video_mxij, video_masks_y=video_myij, mask_index=i
                    )
                    video_outs[i].append(video_out)

        return audio_outs, video_outs


class AudioPredictorMultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y):
        n = 0
        outs = [[] for _ in x]
        for i, (xi, mxi, myi) in enumerate(zip(x, masks_x, masks_y)):
            for xij, mxij, myij in zip(xi, mxi, myi):
                outs[i] += [self.backbone(xij, mxij, myij)]
                n += 1
        return outs


class MJEPAPredictorHierMultiSeqWrapper(nn.Module):
    """
    Wrapper for hierarchical predictor that returns 4 outputs:
    audio_pred, video_pred, audio_context, video_context
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        audio_x=None,
        audio_masks_x=None,
        audio_masks_y=None,
        video_x=None,
        video_masks_x=None,
        video_masks_y=None,
        mask_index=1,
    ):

        audio_pred_outs = None
        video_pred_outs = None
        audio_context_outs = None
        video_context_outs = None

        if video_x is None and audio_x is None:
            assert False, "No input to the predictor. Both video and audio are none"

        if audio_x is not None and video_x is not None:
            audio_pred_outs = [[] for _ in audio_x]
            video_pred_outs = [[] for _ in video_x]
            audio_context_outs = [[] for _ in audio_x]
            video_context_outs = [[] for _ in video_x]

            for i, (audio_xi, audio_mxi, audio_myi, video_xi, video_mxi, video_myi) in enumerate(
                zip(audio_x, audio_masks_x, audio_masks_y, video_x, video_masks_x, video_masks_y)
            ):
                for audio_xij, audio_mxij, audio_myij, video_xij, video_mxij, video_myij in zip(
                    audio_xi, audio_mxi, audio_myi, video_xi, video_mxi, video_myi
                ):
                    audio_pred, video_pred, audio_context, video_context = self.backbone(
                        audio_x=audio_xij,
                        audio_masks_x=audio_mxij,
                        audio_masks_y=audio_myij,
                        video_x=video_xij,
                        video_masks_x=video_mxij,
                        video_masks_y=video_myij,
                        mask_index=i,
                    )
                    audio_pred_outs[i].append(audio_pred)
                    video_pred_outs[i].append(video_pred)
                    audio_context_outs[i].append(audio_context)
                    video_context_outs[i].append(video_context)

        elif audio_x is not None and video_x is None:
            audio_pred_outs = [[] for _ in audio_x]
            audio_context_outs = [[] for _ in audio_x]

            for i, (audio_xi, audio_mxi, audio_myi) in enumerate(zip(audio_x, audio_masks_x, audio_masks_y)):
                for audio_xij, audio_mxij, audio_myij in zip(audio_xi, audio_mxi, audio_myi):
                    audio_pred, _, audio_context, _ = self.backbone(
                        audio_x=audio_xij, audio_masks_x=audio_mxij, audio_masks_y=audio_myij, mask_index=i
                    )
                    audio_pred_outs[i].append(audio_pred)
                    audio_context_outs[i].append(audio_context)

        elif audio_x is None and video_x is not None:
            video_pred_outs = [[] for _ in video_x]
            video_context_outs = [[] for _ in video_x]

            for i, (video_xi, video_mxi, video_myi) in enumerate(zip(video_x, video_masks_x, video_masks_y)):
                for video_xij, video_mxij, video_myij in zip(video_xi, video_mxi, video_myi):
                    _, video_pred, _, video_context = self.backbone(
                        video_x=video_xij, video_masks_x=video_mxij, video_masks_y=video_myij, mask_index=i
                    )
                    video_pred_outs[i].append(video_pred)
                    video_context_outs[i].append(video_context)

        return audio_pred_outs, video_pred_outs, audio_context_outs, video_context_outs
