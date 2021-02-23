import numpy as np
import numba
import matplotlib.pyplot as plt

from dla_monte_carlo import dla_monte_carlo
from dla_prob_model import dla_prob_model
from gray_scott import gray_scott

def main():

    N = 100             # lattice size
    eta = 1           # eta in dla_prob_model, everything runs quite fast, except eta = 0

    dla_prob_model(N, eta)

    ## TODO: implement monte carlo model
    dla_monte_carlo(N)

    ## TODO: implement Gray-Scott model
    gray_scott()

    return


if __name__ == "__main__":
    main()
