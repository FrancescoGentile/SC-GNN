##
##
##

from __future__ import annotations

import abc

from sbar.utils import create_instance
from .config import Config


class Generator(abc.ABC):

    @staticmethod
    def build(config: Config) -> Generator:
        """
        Create a Generator class based on the passed dataset config.
        :param config:
        :return:
        """
        child_module = config.name
        parent_module = '.'.join(Generator.__module__.split('.')[:-1])
        module = f'{parent_module}.{child_module}'
        class_path = f'{module}.Config'

        return create_instance(class_path, Generator, config)

    @abc.abstractmethod
    def generate(self):
        pass
