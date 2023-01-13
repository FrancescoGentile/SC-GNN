##
##
##

from __future__ import annotations

from typing import Final

import torch.nn as nn
from torchtyping import TensorType

from sbar.datasets import Skeleton
from .config import Config, SpatialConfig
from .spatial import SpatialLayer
from .temporal import TemporalLayer


class SCAT(nn.Module):
    _encoder: Final[Encoder]
    _head: Final[ClassificationHead]

    def __init__(self):
        super(SCAT, self).__init__()

        self._encoder = Encoder()
        self._head = ClassificationHead()

    def forward(self,
                joints: TensorType["batch:-1", "in_channels:-1", "frames:-1", "joints:-1"],
                bones: TensorType["batch:-1", "in_channels:-1", "frames:-1", "bones:-1"],
                skeleton: Skeleton):
        N, M, C, T, J = joints.shape
        B = bones.shape[-1]
        joints = joints.view(N * M, C, T, J)
        bones = bones.view(N * M, C, T, B)
        joints, bones = self._encoder(joints, bones, skeleton)

        _, C, T, _ = joints.shape
        joints = joints.view(N, M, C, T, J)
        bones = joints.view(N, M, C, T, B)
        out = self._head(joints, bones)

        return out


class Encoder(nn.Module):
    _layers: Final[nn.ModuleList]

    def __init__(self, config: Config):
        super(Encoder, self).__init__()

        self._layers = nn.ModuleList()

        for layer_config in config.layers:
            if isinstance(layer_config, SpatialConfig):
                self._layers.append(SpatialLayer(layer_config))
            else:
                self._layers.append(TemporalLayer(layer_config))

    def forward(self,
                joints: TensorType["batch: -1", "in_channels: -1", "frames: -1", "joints: -1"],
                bones: TensorType["batch: -1", "in_channels: -1", "frames: -1", "bones: -1"],
                skeleton: Skeleton):
        pass


class ClassificationHead(nn.Module):
    _fc: Final[nn.Linear]

    def __init__(self):
        super(ClassificationHead, self).__init__()

        self._fc = nn.Linear()

    def forward(self,
                joints: TensorType["batch:-1", "person:-1", "channels:-1", "frames:-1", "joints:-1"],
                bones: TensorType["batch:-1", "person:-1", "channels:-1", "frames:-1", "bones:-1"]) -> \
            TensorType["batch:-1", "actions:-1"]:
        N, M, C, *_ = joints.shape
        joints = joints.view(N, M, C, -1).mean(-1).mean(1)
        bones = bones.view(N, M, C, -1).mean(-1).mean(1)

        tmp = joints + bones
        out = self._fc(tmp)

        return out
