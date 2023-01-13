##
##
##

from __future__ import annotations

import abc
from typing import Tuple

from torchtyping import TensorType

from sbar.utils import create_instance
from .config import Config


class Skeleton(abc.ABC):

    @staticmethod
    def build(config: Config) -> Skeleton:
        """
        Create a Skeleton class based on the passed dataset config.
        :param config:
        :return:
        """
        child_module = config.name
        parent_module = '.'.join(Skeleton.__module__.split('.')[:-1])
        module = f'{parent_module}.{child_module}'
        class_path = f'{module}.Config'

        return create_instance(class_path, Skeleton, config)

    @abc.abstractmethod
    def bones(self) -> TensorType["bones: -1", 2, int]:
        """
        Returns a list of all the bones defined as a tuple with the index of the source joint and the index of the
        target joint.

        :return:
        """
        pass

    @abc.abstractmethod
    def all_pairs_shortest_path(self, joints: bool = True) -> \
            Tuple[TensorType["nodes: -1", "nodes: -1", "path: -1", int],
                  TensorType["nodes: -1", int],
                  int]:
        """
        :param joints:
        :return:
        """
        pass
