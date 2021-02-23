import numpy as np
import numba
import matplotlib.pyplot as plt


def discrete_diffusion(c_u, c_v, dt, dx, f, k, D_u, D_v):
    """Discrete diffusion solved time dependently

    Args:
        c_u (2D nparray): concentration of u at each coordinate
        c_v (2D nparray): concentration of v at each coordinate

    Returns:
        c_u (2D nparray): concentration of u at each coordinate
        c_v (2D nparray): concentration of v at each coordinate

    """
    c_diff_u = (dt*D_u/(dx**2))
    c_diff_v = (dt*D_v/(dx**2))

    return c_u, c_v


@numba.njit
def discrete_diffusion_step(c, c_diffusion):
    """Discrete diffusion solved time dependently for a single time step

    Args:
        c (2D nparray): concentration at each coordinate

    Returns:
        c_new (2D nparray): new concentration at each coordinate

    """
    c_new = np.copy(c)

    for i in range(0, len(c)):
        for j in range(1, len(c[0])-1):
            c_new[i, j] = c[i, j] + c_diffusion * (c[(i+1)%len(c), j] + c[(i-1), j] \
                                        + c[i, j+1] + c[i, j-1] - 4 \
                                        * c[i, j])

    return c_new


def gray_scott(N):
    """Gray-Scott model

    Args:
        N (int): lattice size

    """
    print("Gray-Scott model with N = {}.".format(N))

    dt = 1
    dx = 1
    D_u = 0.16
    D_v = 0.08
    f = 0.035
    k = 0.060

    c_u = np.zeros((N, N))
    c_v = np.zeros((N, N))

    c_u, c_v = discrete_diffusion(c_u, c_v, dt, dx, f, k, D_u, D_v)

    print("")

    return
