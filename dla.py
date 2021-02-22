import numpy as np
import numba
import matplotlib.pyplot as plt


@numba.njit
def sor(c, objects, omega=1.7):
    """Successive Over Relaxation """
    c_new = np.copy(c)
    delta = np.inf
    eps = 1e-5
    while delta > eps:
        for i in range(len(c)):
            for j in range(1, len(c[0])-1):
                if object[i, j] == 1:
                    continue
                c_new[i, j] = (omega/4) * (c_new[(i+1)%len(c), j] + c_new[i-1, j] \
                            + c_new[i, j+1] + c_new[i, j-1]) + (1 - omega) * c_new[i, j]

        delta = np.max(np.abs(c_new - c))
        c = np.copy(c_new)
    return c_new


def main():

    N = 100

    # start with analytical solution for domain, with at y = 1, c = 1
    c = np.array([[j/(N-1) for j in range(N)] for i in range(N)])

    # start with a single object (object has c = 0)
    objects = np.zeros((N, N))
    objects[N//2][0] = 1
    c[N//2][0] = 0

    c_new = sor(c, objects, omega=1.8)

    plt.matshow(c_new)
    plt.colorbar()
    plt.show()

    return


if __name__ == "__main__":
    main()
