import numpy as np
import numba
import matplotlib.pyplot as plt


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


def gray_scott():
    """Gray-Scott model """
    print("Gray-Scott model.")

    return
