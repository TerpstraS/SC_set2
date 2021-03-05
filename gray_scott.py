import numpy as np
import numba
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial


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

    # for i in range(t_total):
    #     c_u_new = discrete_diffusion_step(c_u, c_v, c_diff_u, f_u, f, k)
    #     c_v_new = discrete_diffusion_step(c_v, c_u, c_diff_v, f_v, f, k)
    #
    #     c_u = c_u_new
    #     c_v = c_v_new

    delta = np.inf
    eps = 1e-6
    n_iters = 0
    while delta > eps:
        n_iters += 1
        c_u_new = discrete_diffusion_step(c_u, c_v, c_diff_u, f_u, f, k)
        c_v_new = discrete_diffusion_step(c_v, c_u, c_diff_v, f_v, f, k)

        delta = np.max(np.abs(c_u_new - c_u))

        c_u = c_u_new
        c_v = c_v_new

        if n_iters > 1e5:
            print("Maximum time limit reached!")
            return c_u, c_v

    return c_u, c_v


def discrete_diffusion_animation(c_u, c_v, t_total, dt, dx, f, k, D_u, D_v):
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

    fig, ax = plt.subplots()
    plt.rcParams.update({"font.size": 14})
    ax.matshow(c_u)
    for i in range(t_total):
        if i % 100 == 0:
            plt.cla()
            ax.matshow(c_v)
            plt.pause(0.001)

        c_u_new = discrete_diffusion_step(c_u, c_v, c_diff_u, f_u, f, k)
        c_v_new = discrete_diffusion_step(c_v, c_u, c_diff_v, f_v, f, k)

        c_u = c_u_new
        c_v = c_v_new

    return c_u, c_v


def discrete_diffusion_animation_save(c_u, c_v, t_total, dt, dx, f, k, D_u, D_v):
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
    c_u = c_u
    c_v = c_v
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, len(c_u)), ylim=(0, len(c_u)))
    matrix = ax.matshow(c_u)

    # initialization function: plot the background of each frame
    def init():
        matrix.set_data(c_u)
        return matrix,

    # animation function.  This is called sequentially
    def animate(i, c_u_init, c_v_init):
        global c_u
        global c_v
        if i == 0:
            c_u = c_u_init
            c_v = c_v_init

        c_u_new = discrete_diffusion_step(c_u, c_v, c_diff_u, f_u, f, k)
        c_v_new = discrete_diffusion_step(c_v, c_u, c_diff_v, f_v, f, k)

        c_u = c_u_new
        c_v = c_v_new

        matrix.set_data(c_u)
        return matrix,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, func=partial(animate, c_u_init=c_u, c_v_init=c_v), init_func=init,
                               frames=t_total, interval=1, blit=True, repeat=False)

    # plt.show()
    print("Saving animation...")
    file_name = "results/gray_scott/gray_scott_anim.mp4"
    writervideo = animation.FFMpegWriter(fps=60)
    anim.save(file_name, writer=writervideo)
    print("Done! Animation saved.")

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

    for i in range(0, len(c_1)-0):
        for j in range(0, len(c_1[0])-0):
            c_new[i, j] = f_func(c_1[i, j], c_2[i, j], f, k) + c_1[i, j] \
                        + c_diffusion * (c_1[(i+1)%len(c_1), j] + c_1[(i-1), j] \
                        + c_1[i, (j+1)%len(c_1)] + c_1[i, j-1] - 4 * c_1[i, j])

    return c_new


def run_gray_scott(N, size_square_v, dt, dx, f, k, D_u, D_v, t_total=5000, noise_stddev=None, anim=False, verbose=False, plot_init=False):
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
        anim (boolean): animate diffusion
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
        # c_u[:, N-1], c_u[:, 0], c_u[0, :], c_u[N-1,:] = 0.5, 0.5, 0.5, 0.5

    # plot initial concentrations
    if plot_init:
        plt.matshow(c_u)
        plt.title("Initial concentration of U\n")
        plt.colorbar()

        plt.matshow(c_v)
        plt.title("Initial concentration of V\n")
        plt.colorbar()

    if anim:
        c_u, c_v = discrete_diffusion_animation(c_u, c_v, t_total, dt, dx, f, k, D_u, D_v)
    else:
        c_u, c_v = discrete_diffusion(c_u, c_v, t_total, dt, dx, f, k, D_u, D_v)

    return c_u, c_v


def gray_scott():
    """Gray-Scott model """
    N = 100
    t_total = 15000
    noise_stddev = None # 0.01
    size_square_v = 30

    # default parameters
    dt = 1
    dx = 1
    f = 0.035   # 0.060
    k = 0.060
    D_u = 0.16
    D_v = 0.08

    # perform algorithm
    time_start = time.time()
    c_u, c_v = run_gray_scott(N, size_square_v, dt, dx, f, k, D_u, D_v, t_total=t_total,
                              noise_stddev=noise_stddev, anim=False, verbose=False, plot_init=False)
    print("Gray-Scott model simulation time: {:.2f} seconds.\n".format(time.time() - time_start))

    # plot final concentrations
    plt.matshow(c_u)
    plt.title("Final concentration of U\n")
    plt.colorbar()
    # plt.savefig("./results/gray_scott/gray_scott_c_u.png")

    plt.matshow(c_v)
    plt.title("Final concentration of V\n")
    plt.colorbar()
    # plt.savefig("./results/gray_scott/gray_scott_c_v.png")

    plt.matshow(c_u+c_v)
    plt.title("Total concentration of $U+V$\n")
    plt.colorbar()
    plt.show()
    return


def gray_scott_experiments():

    N = 100
    t_total = 15000
    noise_stddev = None # 0.01
    size_square_v = 30

    # default parameters
    dt = 1
    dx = 1
    f = 0.035   # 0.060
    k = 0.060
    D_u = 0.16
    D_v = 0.08

    # perform algorithm
    time_start = time.time()
    c_u, c_v = run_gray_scott(N, size_square_v, dt, dx, f, k, D_u, D_v, t_total=t_total,
                              noise_stddev=noise_stddev, anim=False, verbose=False, plot_init=False)
    print("Gray-Scott model simulation time: {:.2f} seconds.\n".format(time.time() - time_start))

    # plot final concentrations
    plt.matshow(c_u)
    plt.title("Final concentration of U\n")
    plt.colorbar()
    # plt.savefig("./results/gray_scott/gray_scott_c_u.png")

    plt.matshow(c_v)
    plt.title("Final concentration of V\n")
    plt.colorbar()
    # plt.savefig("./results/gray_scott/gray_scott_c_v.png")

    plt.matshow(c_u+c_v)
    plt.title("Total concentration of $U+V$\n")
    plt.colorbar()
    plt.show()

    dt = 1
    dx = 1
    f = 0.035   # 0.060
    k = 0.060
    D_u = 0.16
    D_v = 0.08

    c_u, c_v = run_gray_scott(N, size_square_v, dt, dx, f, k, D_u, D_v, t_total=t_total,
                              noise_stddev=noise_stddev, anim=False, verbose=False, plot_init=False)

    fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
    plt.rcParams.update({"font.size": 14})
    im = None
    # fs = [0.03, 0.03, 0.03, 0.03]
    # ks = [0.055, 0.055, 0.055, 0.055]
    # D_us = [0.15, 0.15, 0.15, 0.15]
    # D_vs = [0.08, 0.08, 0.08, 0.08]
    fs = [0.035, 0.035, 0.035, 0.035]
    ks = [0.060, 0.060, 0.060, 0.060]
    D_us = [0.2, 0.2, 0.2, 0.2]
    D_vs = [0.09, 0.09, 0.09, 0.09]
    for i, ax in enumerate(axs.flatten()):
        if i == 0:
            c_u, c_v = run_gray_scott(N, size_square_v, dt, dx, fs[i], ks[i], D_us[i], D_vs[i], t_total=t_total,
                                      noise_stddev=None, anim=False, verbose=False, plot_init=False)
            im = ax.matshow(c_u.T, origin="lower", extent=[0., 1., 0., 1.])
            ax.set_title("U")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.1)
            fig.colorbar(im, cax=cax, orientation="vertical")
        elif i == 1:
            c_u, c_v = run_gray_scott(N, size_square_v, dt, dx, fs[i], ks[i], D_us[i], D_vs[i], t_total=t_total,
                                      noise_stddev=None, anim=False, verbose=False, plot_init=False)
            im = ax.matshow(c_v.T, origin="lower", extent=[0., 1., 0., 1.])
            ax.set_title("V")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.1)
            fig.colorbar(im, cax=cax, orientation="vertical")
        if i == 2:
            c_u, c_v = run_gray_scott(N, size_square_v, dt, dx, fs[i], ks[i], D_us[i], D_vs[i], t_total=t_total,
                                      noise_stddev=0.05, anim=False, verbose=False, plot_init=False)
            im = ax.matshow(c_u.T, origin="lower", extent=[0., 1., 0., 1.])
            ax.set_title("U noise")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.1)
            fig.colorbar(im, cax=cax, orientation="vertical")
        elif i == 3:
            c_u, c_v = run_gray_scott(N, size_square_v, dt, dx, fs[i], ks[i], D_us[i], D_vs[i], t_total=t_total,
                                      noise_stddev=0.05, anim=False, verbose=False, plot_init=False)
            im = ax.matshow(c_v.T, origin="lower", extent=[0., 1., 0., 1.])
            ax.set_title("V noise")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.1)
            fig.colorbar(im, cax=cax, orientation="vertical")

    # fig.colorbar(im, ax=axs.ravel().tolist())
    axs[1, 0].xaxis.tick_bottom()
    axs[1, 0].set_xlabel("x")
    axs[1, 1].xaxis.tick_bottom()
    axs[1, 1].set_xlabel("x")
    axs[1, 0].yaxis.tick_left()
    axs[1, 0].set_ylabel("y")
    axs[0, 0].yaxis.tick_left()
    axs[0, 0].set_ylabel("y")
    fig.tight_layout(pad=0)
    plt.savefig("results/gray_scott/varying_init_not_default2.png")

    plt.show()

    return


if __name__ == '__main__':
    gray_scott_experiments()
