##
##
##

from __future__ import annotations

from typing import Dict, Any, Final, Optional


class TrainingConfig:
    _self_supervised: Final[Optional[SelfSupervisedConfig]]
    _supervised: Final[Optional[SupervisedConfig]]

    def __init__(self, cfg: Dict[str, Any]):
        if cfg['self_supervised_args'] is not None:
            self._self_supervised = SelfSupervisedConfig()

        if cfg['supervised_args'] is not None:
            self._supervised = SupervisedConfig(cfg)


class SelfSupervisedConfig:
    pass


class SupervisedConfig:
    _train_batch_size: Final[int]
    _eval_batch_size: Final[int]
    _accumulation_steps: Final[int]
    _num_epochs: Final[int]

    _seed: Final[int] = 0

    def __init__(self, cfg: Dict[str, Any]):
        self._train_batch_size = int(cfg['train_batch_size'])
        self._eval_batch_size = int(cfg['eval_batch_size'])
        self._accumulation_steps = int(cfg['accumulation_steps'])
        self._num_epochs = int(cfg['num_epochs'])

        self._seed = int(cfg['seed'])
