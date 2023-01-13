##
##
##

from pathlib import Path
from typing import Final, List, Tuple

from . import utils
from .config import Config, GenerationConfig, Benchmark
from .. import Generator as AbstractGenerator


class Generator(AbstractGenerator):
    _config: Final[GenerationConfig]
    _benchmark: Final[Benchmark]

    def __init__(self, config: Config):
        super().__init__()

        self._benchmark = config.benchmark

        if config.generation is None:
            raise ValueError('Missing generation args for NTU.')
        self._config = config.generation

    def _is_training_file(self, file: str) -> bool:
        match self._benchmark:
            case Benchmark.NTU60_XSUB | Benchmark.NTU120_XSUB:
                return utils.get_subject_id(file) in {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34,
                                                      35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70,
                                                      74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97,
                                                      98, 100, 103}
            case Benchmark.NTU60_XVIEW:
                return utils.get_camera_id(file) in {2, 3}
            case Benchmark.NTU120_XSET:
                return utils.get_setup_id(file) in set(range(2, 33, 2))
            case _:
                raise ValueError(f'Missing training ids for benchmark {self._benchmark}.')

    def _train_test_files(self) -> Tuple[List[Path], List[Path]]:
        """
        Returns the lists of the files for training and testing.
        :return:
        """
        train = []
        test = []

        with self._config.ignore_path.open('r') as f:
            ignore_files = set([f'{line.strip()}.skeleton' for line in f.readlines()])

        for folder in [self._config.ntu60_path, self._config.ntu120_path]:
            if folder is None:
                continue

            for file in folder.iterdir():
                if file.is_file() and file.name not in ignore_files:
                    if self._is_training_file(file.name):
                        train.append(file)
                    else:
                        test.append(file)

        return train, test

    def generate(self):
        train, test = self._train_test_files()
