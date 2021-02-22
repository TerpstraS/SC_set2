import numpy as np
import numba
import matplotlib.pyplot as plt


def dla_prob_model(N, eta):
    """DLA model using growth probability

    Args:
        N (int): lattice size
    """

    n = 100   # total simulation steps

    # start with analytical solution for domain, with at y = 1, c = 1
    c = np.array([[j/(N-1) for j in range(N)] for i in range(N)])

    # start with a single object (object has c = 0)
    objects = np.zeros((N, N))
    objects[N//2][0] = 1
    c[N//2][0] = 0

    # perform simulation
    for i in range(n):

        # perform sor iteration to determine new nutrient field
        c = sor(c, objects, omega=1.8)

        # determine growth candidates

        # calculate growth probabilities

        # choose growth location and add location to objects

    plt.matshow(c)
    plt.colorbar()
    plt.show()
    return


@numba.njit
def sor(c, objects, omega=1.7):
    """Successive Over Relaxation """
    c_new = np.copy(c)
    delta = np.inf
    eps = 1e-5
    while delta > eps:
        for i in range(len(c)):
            for j in range(1, len(c[0])-1):
                if objects[i, j] == 1:
                    continue
                c_new[i, j] = (omega/4) * (c_new[(i+1)%len(c), j] + c_new[i-1, j] \
                            + c_new[i, j+1] + c_new[i, j-1]) + (1 - omega) * c_new[i, j]

        delta = np.max(np.abs(c_new - c))
        c = np.copy(c_new)
    return c_new
