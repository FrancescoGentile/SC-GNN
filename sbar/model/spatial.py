##
##
##

import torch
import torch.nn as nn
from torchtyping import TensorType

from sbar.datasets import Skeleton
from .config import SpatialConfig


class SpatialLayer(nn.Module):

    def __init__(self, config: SpatialConfig):
        super(SpatialLayer, self).__init__()

        self._att_layer = JointAttentionLayer(config)
        self._bone_layer = BoneLayer(config)

    def forward(self,
                joints: TensorType["batch: -1", "in_channels: -1", "frames: -1", "joints: -1"],
                bones: TensorType["batch: -1", "in_channels: -1", "frames: -1", "bones: -1"],
                skeleton: Skeleton):
        joints = self._att_layer(joints, bones, skeleton)
        bones = self._bone_layer(joints, bones, skeleton)

        return joints, bones


class JointAttentionLayer(nn.Module):

    def __init__(self, config: SpatialConfig):
        super(JointAttentionLayer, self).__init__()

        self._att_mlp = nn.Sequential(
            nn.Conv2d(config.in_channels * 3, config.in_channels, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(config.in_channels, 1, kernel_size=(1, 1))
        )

        self._transform = nn.Conv2d(config.in_channels * 3, config.out_channels, kernel_size=(1, 1))

    def forward(self,
                joints: TensorType["batch: -1", "in_channels: -1", "frames: -1", "joints: -1", float],
                bones: TensorType["batch: -1", "in_channels: -1", "frames: -1", "bones: -1", float],
                skeleton: Skeleton) -> TensorType["batch: -1", "out_channels: -1", "frames: -1", "joints: -1", float]:
        N, C, T, J = joints.shape
        source = torch.repeat_interleave(joints, J, dim=-1)
        target = torch.repeat_interleave(joints, J, dim=0)
        target = target.view(N, C, T, -1)

        bone_paths = self._compute_path(bones, joints_path=False, skeleton=skeleton)
        attn_input = torch.cat((source, target, bone_paths), dim=1)
        attn_logits = self._att_mlp(attn_input)
        attn_logits = attn_logits.view(N, 1, T, J, J)
        attn_weights = torch.softmax(attn_logits, dim=-1)

        joint_paths = self._compute_path(joints, joints_path=True, skeleton=skeleton)
        value_input = torch.cat((source, target, joint_paths), dim=1)
        values = self._transform(value_input)  # (B, C_out, T, J * J)

        out = values * attn_weights
        out = torch.sum(out, dim=-1)

        return TensorType(out)

    @staticmethod
    def _compute_path(x: TensorType[...], joints_path: bool, skeleton: Skeleton) -> TensorType[...]:
        paths, paths_length, idx = skeleton.all_pairs_shortest_path(joints=joints_path)
        max_path_length = paths.shape[-1]

        x_paths = x[..., paths]
        x_paths = x_paths.sum(-1)
        x_paths = x_paths - (max_path_length - paths_length) * x[..., idx, None]
        x_paths = x_paths / paths_length

        return x_paths


class BoneLayer(nn.Module):

    def __init__(self, config: SpatialConfig):
        super(BoneLayer, self).__init__()

        self._transform = nn.Conv2d(config.out_channels * 2 + config.in_channels,
                                    config.out_channels,
                                    kernel_size=(1, 1))

    def forward(self,
                joints: TensorType["batch: -1", "out_channels: -1", "frames: -1", "joints: -1", float],
                bones: TensorType["batch: -1", "in_channels: -1", "frames: -1", "bones: -1", float],
                skeleton: Skeleton) -> TensorType["batch: -1", "out_channels: -1", "frames: -1", "bones: -1", float]:
        edges = skeleton.bones()
        source = joints[..., edges[:, 0]]
        target = joints[..., edges[:, 1]]

        value_input = torch.cat((bones, source, target), dim=1)
        values = self._transform(value_input)

        return values
