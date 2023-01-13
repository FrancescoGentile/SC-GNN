##
##
##

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sbar.datasets.ntu.sequence


def main():
    file = Path('/home/francesco/Documents/university/skeleton/nturgb+d_skeletons/S009C002P025R002A010.skeleton')
    data = np.zeros((2, 64, 25, 3))
    sequence = sbar.datasets.ntu.sequence.ActionSequence.from_file(file, data)

    plt.ion()
    fig = plt.figure(figsize=(200, 200))
    ax = fig.add_subplot(111, projection='3d')

    sequence.display(ax)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
