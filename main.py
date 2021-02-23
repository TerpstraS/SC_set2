import numpy as np
import numba
import matplotlib.pyplot as plt

from dla_monte_carlo import dla_monte_carlo
from dla_prob_model import dla_prob_model


def main():

    N = 100             # lattice size
    n_steps = 100       # number of simulation steps
    delta = 1           # eta in dla_prob_model

    dla_prob_model(N, n_steps, delta)
    # dla_monte_carlo(N)

    return


if __name__ == "__main__":
    main()
