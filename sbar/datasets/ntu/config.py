##
##
##

from __future__ import annotations

import enum
from pathlib import Path
from typing import Dict, Any, Final, Optional

from .. import Config as AbstractConfig


class Config(AbstractConfig):
    _path: Final[Path]
    _benchmark: Final[Benchmark]

    _generation: Final[Optional[GenerationConfig]] = None
    _self_supervised: Final[Optional[SelfSupervisedConfig]] = None
    _supervised: Final[Optional[SupervisedConfig]] = None

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

        self._path = Path(cfg['path'])

        benchmark = str(cfg['benchmark']).upper().replace('-', '_')
        self._benchmark = Benchmark[benchmark]

        if cfg['generation_args'] is not None:
            self._generation = GenerationConfig(cfg['generation_args'], self._benchmark)

        if cfg['self_supervised_args'] is not None:
            self._self_supervised = SelfSupervisedConfig(cfg['self_supervised_args'])

        if cfg['supervised_args'] is not None:
            self._supervised = SupervisedConfig(cfg['supervised_args'])

    @property
    def benchmark(self) -> Benchmark:
        return self._benchmark

    @property
    def generation(self) -> Optional[GenerationConfig]:
        return self._generation


class Benchmark(enum.Enum):
    NTU60_XSUB = 'ntu60-xsub'
    NTU60_XVIEW = 'ntu60-xview'
    NTU120_XSUB = 'ntu120-xsub'
    NTU120_XSET = 'ntu120-xset'

    def __str__(self):
        return self.value

    @property
    def num_classes(self) -> int:
        match self:
            case Benchmark.NTU60_XSUB | Benchmark.NTU60_XVIEW:
                return 60
            case _:
                return 120

    def is_ntu120(self) -> bool:
        match self:
            case Benchmark.NTU60_XSUB | Benchmark.NTU60_XVIEW:
                return False
            case _:
                return True


class GenerationConfig:
    _ntu60_path: Final[Path]
    _ntu120_path: Final[Optional[Path]] = None
    _ignore_path: Final[Path]

    _num_people: Final[int]
    _num_frames: Final[int]

    def __init__(self, cfg: Dict[str, Any], benchmark: Benchmark):
        if cfg['ntu60_path'] is None:
            raise ValueError(f'Missing \'ntu60_path\' is generation args.')
        else:
            self._ntu60_path = Path(cfg['ntu60_path'])
            if not self._ntu60_path.is_dir():
                raise ValueError(f'{self._ntu60_path} is not a directory.')

        if benchmark.is_ntu120():
            if cfg['ntu120_path'] is None:
                raise ValueError(f'Missing \'ntu120_path\' in generation args.')
            else:
                self._ntu120_path = Path(cfg['ntu120_path'])
                if not self._ntu120_path.is_dir():
                    raise ValueError(f'{self._ntu120_path} is not a directory.')

        if cfg['ignore_path'] is None:
            raise ValueError(f'Missing \'ignore_path\' in generation args.')
        else:
            self._ignore_path = Path(cfg['ignore_path'])
            if not self._ignore_path.is_file():
                raise ValueError(f'{self._ignore_path} is not a file.')

        if cfg['num_people'] is not None:
            self._num_people = int(cfg['num_people'])

        if cfg['num_frames'] is not None:
            self._num_frames = int(cfg['num_frames'])

    @property
    def ntu60_path(self) -> Path:
        """
        Path to the directory containing the samples common to both
        NTU RGB+D 60 and NTU RGB+D 120.
        """
        return self._ntu60_path

    @property
    def ntu120_path(self) -> Optional[Path]:
        """
        Path to the directory containing the samples added in NTU RGB+D 120.
        """
        return self._ntu120_path

    @property
    def ignore_path(self) -> Path:
        """
        Path to the file with the list of samples to be ignored.
        """
        return self._ignore_path

    @property
    def num_people(self) -> int:
        return self._num_people

    @property
    def num_frames(self) -> int:
        return self._num_frames


class SelfSupervisedConfig:
    _standardize: Final[bool] = False

    def __init__(self, cfg: Dict[str, Any]):
        if cfg['standardize'] is not None:
            self._standardize = bool(cfg['standardize'])

    @property
    def standardize(self) -> bool:
        """
        Whether to standardize the joint coordinates or not.
        """
        return self._standardize


class SupervisedConfig:
    _standardize: Final[bool] = False

    def __init__(self, cfg: Dict[str, Any]):
        if cfg['standardize'] is not None:
            self._standardize = bool(cfg['standardize'])

    @property
    def standardize(self) -> bool:
        """
        Whether to standardize the joint coordinates or not.
        """
        return self._standardize
