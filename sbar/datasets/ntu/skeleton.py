##
##
##

import itertools
from typing import List, Tuple, Final

import networkx as nx
import numpy as np
import torch
from torchtyping import TensorType

from .. import Skeleton as AbstractSkeleton


class Skeleton(AbstractSkeleton):
    _bones: Final[TensorType["bones: -1", 2, int]]

    _padding_value: Final[int] = 0

    _joint_paths: Final[TensorType["nodes: -1", "nodes: -1", "path: -1", int]]
    _joint_paths_len: Final[TensorType["nodes: -1", int]]

    _bone_paths: Final[TensorType["nodes: -1", "nodes: -1", "path: -1", int]]
    _bone_paths_len: Final[TensorType["nodes: -1", int]]

    def __init__(self):
        super(Skeleton, self).__init__()

        bones = [
            (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
            (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
            (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
            (19, 18), (20, 19), (8, 22), (7, 23), (12, 24), (11, 25)]
        graph = build_graph(num_nodes=25, edges=bones, self_loops=True)

        self._bones = torch.tensor(bones) - torch.tensor([1, 1])

        self._joint_paths, self._joint_paths_len = compute_joint_paths(graph,
                                                                       keep_source=False,
                                                                       keep_target=False,
                                                                       padding_value=self._padding_value)

        self._bone_paths, self._bone_paths_len = compute_bone_paths(graph,
                                                                    bones=self._bones,
                                                                    padding_value=self._padding_value)

    def bones(self) -> TensorType["bones: -1", 2, int]:
        return self._bones

    def all_pairs_shortest_path(self, joints: bool = True) -> \
            Tuple[TensorType["nodes: -1", "nodes: -1", "path: -1", int],
                  TensorType["nodes: -1", int],
                  int]:
        if joints:
            return self._joint_paths, self._joint_paths_len, self._padding_value
        else:
            return self._bone_paths, self._bone_paths_len, self._padding_value

    @staticmethod
    def bones_pairs() -> np.ndarray:
        bones = np.array([
            (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
            (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
            (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
            (19, 18), (20, 19), (8, 22), (7, 23), (12, 24), (11, 25)])
        bones = bones - (1, 1)

        return bones


def build_graph(num_nodes: int, edges: List[Tuple[int, int]], self_loops: bool = True) -> nx.Graph:
    graph = nx.Graph(edges)
    if self_loops:
        self_edges = [(i, i) for i in range(num_nodes)]
        graph.add_edges_from(self_edges)

    return graph


def compute_joint_paths(graph: nx.Graph,
                        keep_source: bool = False,
                        keep_target: bool = False,
                        padding_value: int = 0) -> \
        Tuple[TensorType["nodes: -1", "nodes: -1", "path: -1", int], TensorType["nodes: -1", int]]:
    full_paths = dict(nx.all_pairs_shortest_path(graph))
    paths = []
    paths_length = []

    nodes = graph.number_of_nodes()
    for u, v in itertools.product(range(nodes), range(nodes)):
        path: List[int] = full_paths[u][v].copy()
        if not keep_source:
            del path[0]
        if not keep_target:
            del path[-1]

        paths.append(torch.tensor(path))
        paths_length.append(len(path))

    paths = torch.nn.utils.rnn.pad_sequence(paths, batch_first=True, padding_value=padding_value)
    paths_length = torch.tensor(paths_length)

    return TensorType(paths), TensorType(paths_length)


def compute_bone_paths(graph: nx.Graph, bones: TensorType[...], padding_value: int = 0) -> \
        Tuple[TensorType["nodes: -1", "nodes: -1", "path: -1", int], TensorType["nodes: -1", int]]:
    full_paths = dict(nx.all_pairs_shortest_path(graph))
    paths = []
    paths_length = []

    nodes = graph.number_of_nodes()
    for u, v in itertools.product(range(nodes), range(nodes)):
        path: List[int] = full_paths[u][v]
        bone_path = []
        for i, j in zip(path[:-1], path[1:]):
            idx = torch.where((bones == torch.tensor([i, j])).all(dim=1))[0]
            bone_path.append(idx.item())

        paths.append(torch.tensor(bone_path))
        paths_length.append(len(bone_path))

    paths = torch.nn.utils.rnn.pad_sequence(paths, batch_first=True, padding_value=padding_value)
    paths_length = torch.tensor(paths_length)

    return TensorType(paths), TensorType(paths_length)
