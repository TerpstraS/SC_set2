import numpy as np
import numba
import matplotlib.pyplot as plt

from dla_monte_carlo import dla_monte_carlo
from dla_prob_model import dla_prob_model
from gray_scott import gray_scott


def main():

    N = 100      # lattice size
    dla_prob_model()
    grid = dla_monte_carlo(N)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.matshow(grid.T, origin="lower", extent=[0., 1., 0., 1.])
    ax.xaxis.tick_bottom()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.yaxis.tick_left()
    plt.savefig("DLA monte carlo 1")
    plt.show()
    gray_scott()

    return


if __name__ == "__main__":
    main()
