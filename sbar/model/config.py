##
##
##

from __future__ import annotations

from typing import Dict, Any, List, Union, Final


class Config:
    _layers: Final[List[Union[SpatialConfig, TemporalConfig]]]

    def __init__(self, cfg: Dict[str, Any]):
        self._layers = []

    def to_dict(self) -> Dict[str, Any]:
        layers = []
        for layer in self._layers:
            layers.append(layer.to_dict())

        return {'layers': layers}

    @property
    def layers(self) -> List[Union[SpatialConfig, TemporalConfig]]:
        return self._layers


class SpatialConfig:
    _in_channels: Final[int]
    _out_channels: Final[int]

    def __init__(self, in_channels: int, cfg: Dict[str, Any]):
        self._in_channels = in_channels
        self._out_channels = int(cfg['out_channels'])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'in_channels': self._in_channels,
            'out_channels': self._out_channels
        }

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels


class TemporalConfig:
    _in_channels: Final[int]
    _out_channels: Final[int]
    _window: Final[int]
    _stride: Final[int] = 1
    _dilation: Final[int] = 1

    def __init__(self, in_channels: int, cfg: Dict[str, Any]):
        self._in_channels = in_channels
        self._out_channels = int(cfg['out_channels'])
        self._window = int(cfg['window'])

        if cfg['stride'] is not None:
            self._stride = int(cfg['stride'])

        if cfg['dilation'] is not None:
            self._dilation = int(cfg['dilation'])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'out_channels': self._out_channels,
            'window': self._window,
            'stride': self._stride,
            'dilation': self._dilation
        }

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def window(self) -> int:
        return self._window

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def dilation(self) -> int:
        return self._dilation
