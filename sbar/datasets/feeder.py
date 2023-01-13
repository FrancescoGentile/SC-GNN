##
##
##

from __future__ import annotations

import abc

from torch.utils.data import Dataset

from sbar.utils import create_instance
from .config import Config


class Feeder(Dataset, abc.ABC):

    @staticmethod
    def build(config: Config) -> Feeder:
        """
        Create a Feeder class based on the passed dataset config.
        :param config:
        :return:
        """
        child_module = config.name
        parent_module = '.'.join(Feeder.__module__.split('.')[:-1])
        module = f'{parent_module}.{child_module}'
        class_path = f'{module}.Config'

        return create_instance(class_path, Feeder, config)
