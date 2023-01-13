##
##
##

from .config import Config
from .. import Feeder as AbstractFeeder


class Feeder(AbstractFeeder):

    def __init__(self, _config: Config):
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError
