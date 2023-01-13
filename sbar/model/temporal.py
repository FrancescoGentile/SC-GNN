##
##
##

from typing import Tuple, Final

import torch
import torch.nn as nn
import torchvision
from torchtyping import TensorType

from .config import TemporalConfig


class DeformableConvolution1D(nn.Module):
    _stride: Final[Tuple[int, int]]
    _padding: Final[Tuple[int, int]]
    _dilation: Final[Tuple[int, int]]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = True):
        super(DeformableConvolution1D, self).__init__()

        kernel_size = (kernel_size, 1)
        self._stride = (stride, 1)
        self._padding = (padding, 1)
        self._dilation = (dilation, 1)

        self.offset_conv = nn.Conv2d(in_channels,
                                     kernel_size[0],
                                     kernel_size=kernel_size,
                                     stride=self._stride,
                                     padding=self._padding,
                                     dilation=self._dilation,
                                     bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.modulation_conv = nn.Conv2d(in_channels,
                                         kernel_size[0],
                                         kernel_size=kernel_size,
                                         stride=self._stride,
                                         padding=self._padding,
                                         dilation=self._dilation,
                                         bias=True)
        nn.init.constant_(self.modulation_conv.weight, 0.0)
        nn.init.constant_(self.modulation_conv.bias, 0.0)

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              bias=bias)

    def forward(self, x: TensorType["batch", "channels", "frames", -1, float]):
        offset = self.offset_conv(x)
        ones = torch.ones_like(offset, requires_grad=False)

        # offset = torch.repeat_interleave(offset, 2, dim=1)
        # offset[:, 1::2] = 0.0
        offset = torch.stack((offset, ones), dim=2)

        modulation = self.modulation_conv(x)
        modulation = torch.sigmoid(modulation)

        out = torchvision.ops.deform_conv2d(x,
                                            offset=offset,
                                            weight=self.conv.weight,
                                            bias=self.conv.bias,
                                            stride=self._stride,
                                            padding=self._padding,
                                            dilation=self._dilation,
                                            mask=modulation)

        return out


class TemporalLayer(nn.Module):

    def __init__(self, config: TemporalConfig):
        super(TemporalLayer, self).__init__()

    def forward(self,
                joints: TensorType["batch", "channels", "frames", "joints", float],
                bones: TensorType["batch", "channels", "frames", "bones", float]):
        pass
