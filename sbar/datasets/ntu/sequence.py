##
##
##

from __future__ import annotations

from pathlib import Path
from typing import Final, Optional, List, Union, Dict

import matplotlib.pyplot as plt
import numpy as np

from . import utils
from .skeleton import Skeleton


class ActionSequence:
    _data: Final[np.ndarray]  # (M, T, J, C)
    _action: Final[int]

    def __init__(self, data: np.ndarray, action: int):
        self._data = data
        self._action = action

    @staticmethod
    def from_file(file: Path, array: np.ndarray) -> ActionSequence:
        action = utils.get_action_id(file.name)

        people = read_file(file)
        max_people, max_frames, _, _ = array.shape
        max_people = max(max_people, len(people))

        energy = [person.get_coords_std() for person in people]
        indexes = np.argsort(energy)[::-1][:max_people]
        people = [people[idx] for idx in indexes]

        people_frames = [person.get_valid_frames() for person in people]
        seq_length = max([max_frames] + [len(frames) for frames in people_frames])
        count = [0 for _ in range(seq_length)]
        for person_frames in people_frames:
            for frame_idx in person_frames:
                count[frame_idx] += 1

        start = max_subsequence(count, max_frames)
        for idx, person in enumerate(people):
            person.interpolate(start, max_frames)
            array[idx] = person.joints

        return ActionSequence(array, action)

    def display(self, ax: plt.Axes):
        people = []
        for idx in range(self._data.shape[0]):
            person = Person.from_joints(self._data[idx])
            if person is not None:
                people.append(person)

        for frame_idx in range(self._data.shape[1]):
            ax.cla()

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            ax.elev = -90
            ax.axim = 61
            ax.roll = 31

            for person in people:
                person.display(ax, frame_idx)

            plt.pause(1)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def action(self) -> int:
        return self._action


class Person:
    _joints: np.ndarray  # (T, J, C)

    def __init__(self, joints: np.ndarray):
        self._joints = joints

    @staticmethod
    def from_joints(joints: np.ndarray) -> Optional[Person]:
        if np.sum(joints) == 0:
            return None
        else:
            return Person(joints)

    @staticmethod
    def from_frames(num_frames: int) -> Person:
        return Person(np.zeros((num_frames, 25, 3), dtype=np.float32))

    def set_coords(self, frame_idx: int, joint_idx: int, coords: List[Union[float, str]]):
        self._joints[frame_idx, joint_idx] = coords

    def get_coords_std(self) -> float:
        indexes = self.get_valid_frames()
        ratio = len(indexes) / self._joints.shape[0]

        std = 0.0
        if len(indexes) != 0:
            joints = self._joints[indexes]
            for i in range(3):
                std += joints[:, :, i].std()

            std *= ratio

        return std

    def get_valid_frames(self) -> List[int]:
        return np.where((np.sum(self._joints, axis=(1, 2)) != 0))[0]

    def interpolate(self, start: int, length: int):
        # if length >= self._joints.shape[0]:
        #    return
        # self._joints = self._joints[:length]
        # return

        _, J, C = self._joints.shape
        new_joints = np.zeros((length, J, C))

        indexes = self.get_valid_frames()
        time_coords = np.arange(start, start + length)
        for joint_idx in range(J):
            for coord_idx in range(C):
                new_joints[:, joint_idx, coord_idx] = np.interp(time_coords, indexes,
                                                                self._joints[indexes, joint_idx, coord_idx])

        self._joints = new_joints

    def display(self, ax: plt.Axes, frame_idx: int):
        x = self._joints[frame_idx, :, 0]
        y = self._joints[frame_idx, :, 1]
        z = self._joints[frame_idx, :, 2]

        for (u, v) in Skeleton.bones_pairs():
            ax.plot3D([x[u], x[v]], [y[u], y[v]], [z[u], z[v]], '-o', c=np.array([0.1, 0.1, 0.1]),
                      linewidth=0.5, markersize=0)

        ax.scatter3D(x, y, z, marker='o', s=16)

    @property
    def joints(self) -> np.ndarray:
        return self._joints


# functions

def read_file(file: Path) -> List[Person]:
    people: Dict[str, Person] = {}
    with file.open('r') as f:
        num_frames = int(f.readline())

        for frame_idx in range(num_frames):
            num_people = int(f.readline())

            for _ in range(num_people):
                person_info = f.readline().strip('\r\n').split()
                person_id = person_info[0]

                if person_id not in people:
                    person = Person.from_frames(num_frames)
                    people[person_id] = person
                else:
                    person = people[person_id]

                num_joints = int(f.readline())
                for joint_idx in range(num_joints):
                    joint_info = f.readline().strip('\r\n').split()
                    person.set_coords(frame_idx, joint_idx, joint_info[:3])

    return list(people.values())


def max_subsequence(seq: List[int], sub_length: int) -> int:
    max_sum = max(seq[0:sub_length])
    start = 0
    for idx in range(sub_length, len(seq)):
        tmp_sum = max_sum + seq[idx] - seq[idx - sub_length]
        if tmp_sum > max_sum:
            max_sum = tmp_sum
            start = idx - sub_length + 1

    return start
