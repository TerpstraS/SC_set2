import numpy as np
import numba
import matplotlib.pyplot as plt
import time


@numba.njit
def f_u(u, v, f, k):
    """Reaction function for chemical agent U """
    return -u * v**2 + f * (1 - u)


@numba.njit
def f_v(v, u, f, k):
    """Reaction function for chemical agent V """
    return u * v**2 - (f + k) * v


@numba.njit
def discrete_diffusion(c_u, c_v, t_total, dt, dx, f, k, D_u, D_v):
    """Discrete diffusion solved time dependently

    Args:
        c_u (2D nparray): concentration of U at each coordinate
        c_v (2D nparray): concentration of V at each coordinate
        t_total (int): total time (# amount of time steps)
        dt (float): dt
        dx (float): dx
        f (float): supply rate of chemical agent U
        k (float): decay rate of chemical agent V
        D_u (float): diffusion constant chemical agent U
        D_v (float): diffusion constant chemical agent V

    Returns:
        c_u (2D nparray): concentration of U at each coordinate
        c_v (2D nparray): concentration of V at each coordinate

    """
    c_diff_u = dt*D_u/(dx**2)
    c_diff_v = dt*D_v/(dx**2)

    for i in range(t_total):
        c_u_new = discrete_diffusion_step(c_u, c_v, c_diff_u, f_u, f, k)
        c_v_new = discrete_diffusion_step(c_v, c_u, c_diff_v, f_v, f, k)

        c_u = c_u_new
        c_v = c_v_new

    return c_u, c_v


@numba.njit
def discrete_diffusion_step(c_1, c_2, c_diffusion, f_func, f, k):
    """Discrete diffusion solved time dependently for a single time step

    Args:
        c_1 (2D nparray): concentration at each coordinate for agent 1 to solve diffusion equation
        c_2 (2D nparray): concentration at each coordinate for agent 2
        c_diffusion (float): constant factor determined by: dt*D_1/(dx**2)
        f_func (function): reaction function for chemical agent 1
        f (float): supply rate of chemical agent U
        k (float): decay rate of chemical agent V

    Returns:
        c_new (2D nparray): new concentration at each coordinate for agent 1

    """
    c_new = np.copy(c_1)

    for i in range(1, len(c_1)-1):
        for j in range(1, len(c_1[0])-1):
            c_new[i, j] = f_func(c_1[i, j], c_2[i, j], f, k) + c_1[i, j] \
                        + c_diffusion * (c_1[(i+1)%len(c_1), j] + c_1[(i-1), j] \
                        + c_1[i, (j+1)%len(c_1)] + c_1[i, j-1] - 4 * c_1[i, j])

    return c_new


def run_gray_scott(N, size_square_v, dt, dx, f, k, D_u, D_v, t_total=5000, noise_stddev=None, verbose=False, plot_init=False):
    """Run the Gray-Scott model for the specified parameters

    Args:
        N (int): lattice size
        size_square_v (int): size of square in center where V has initial concentration
        dt (float): dt
        dx (float): dx
        f (float): supply rate of chemical agent U
        k (float): decay rate of chemical agent V
        D_u (float): diffusion constant chemical agent U
        D_v (float): diffusion constant chemical agent V
        t_total (int): total time (# amount of time steps)
        noise_stddev (float): sttdev for noise added to initial conditions. If None no noise added
        verbose (boolean): display info
        plot_init (boolean): plot initial conditions

    Returns:
        c_u (2D nparray): concentration of U at each coordinate
        c_v (2D nparray): concentration of V at each coordinate

    """
    if verbose:
        print("Running Gray-Scott simulation:\
              \nN\t{},\t\tsize_square_v\t{},\ndt\t{},\t\tdx\t{},\nf\t{},\t\tk\t{},\
              \nD_u\t{},\t\tD_v\t{},\nt_total\t{},\t\tnoise_stddev\t{}\n"
              .format(N, size_square_v, dt, dx, f, k, D_u, D_v, t_total, noise_stddev))

    # create initial concentrations of chemical agents U and V
    c_u = np.full((N, N), 0.5)
    c_v = np.zeros((N, N))
    size_box = 30
    c_v[N//2-N//size_square_v:N//2+N//size_square_v, N//2-N//size_square_v:N//2+N//size_square_v] = 0.25

    # add gaussian noise to initial conditions
    if noise_stddev:
        noise_mean = 0
        c_u += np.random.normal(noise_mean, noise_stddev, (N, N))
        c_v += np.random.normal(noise_mean, noise_stddev, (N, N))

    # plot initial concentrations
    if plot_init:
        plt.matshow(c_u)
        plt.title("Initial concentration of U\n")
        plt.colorbar()

        plt.matshow(c_v)
        plt.title("Initial concentration of V\n")
        plt.colorbar()

    c_u, c_v = discrete_diffusion(c_u, c_v, t_total, dt, dx, f, k, D_u, D_v)

    return c_u, c_v


def gray_scott():
    """Gray-Scott model """
    N = 100
    t_total = 5000
    noise_stddev = 0.01
    size_square_v = 30

    # default parameters
    dt = 1
    dx = 1
    f = 0.035
    k = 0.060
    D_u = 0.16
    D_v = 0.06

    # perform algorithm
    time_start = time.time()
    c_u, c_v = run_gray_scott(N, size_square_v, dt, dx, f, k, D_u, D_v, t_total=t_total,
                              noise_stddev=noise_stddev, verbose=False, plot_init=False)
    print("Gray-Scott model simulation time: {:.2f} seconds.\n".format(time.time() - time_start))

    # plot final concentrations
    plt.matshow(c_u)
    plt.title("Final concentration of U\n")
    plt.colorbar()
    plt.savefig("./results/gray_scott/gray_scott_c_u.png")

    plt.matshow(c_v)
    plt.title("Final concentration of V\n")
    plt.colorbar()
    plt.savefig("./results/gray_scott/gray_scott_c_v.png")
    plt.show()

    return
