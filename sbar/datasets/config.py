##
##
##

from __future__ import annotations

import abc
from typing import final, Final, Dict, Any

from sbar.utils import create_instance


class Config(abc.ABC):
    _name: Final[str]

    def __init__(self, cfg: Dict[str, Any]):
        self._name = str(cfg['name']).lower()

    @property
    @final
    def name(self) -> str:
        return self._name

    @staticmethod
    def build(cfg: Dict[str, Any]) -> Config:
        """
        Create a Config class based on the passed config dictionary.
        :param cfg:
        :return:
        """
        child_module = str(cfg['name']).lower()
        parent_module = '.'.join(Config.__module__.split('.')[:-1])
        module = f'{parent_module}.{child_module}'
        class_path = f'{module}.Config'

        return create_instance(class_path, Config, cfg)
