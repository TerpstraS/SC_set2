import numpy as np
import numba
import matplotlib.pyplot as plt

from dla_monte_carlo import dla_monte_carlo
from dla_prob_model import dla_prob_model
from gray_scott import gray_scott

def main():

    dla_prob_model()

    # TODO: implement monte carlo model
    N = 100      # lattice size
    dla_monte_carlo(N)

    # Gray-Scott model
    gray_scott()

    return


if __name__ == "__main__":
    main()
