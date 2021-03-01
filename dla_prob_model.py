import numpy as np
import numba
import time
import matplotlib.pyplot as plt
import tqdm


@numba.njit
def get_growth_candidates(N, c, objects):
    """Find all growth candidates i.e. all empty lattice points north, east, south west of each
    object that are free and have non-zero concentration

    Args:
        N (int): lattice size
        c (2D nparray): concentration at each coordinate
        objects (2D nparray): array of whole grid space. Value of 1 means object, 0 means no object

    Returns:
        candidates (2D nparray): array of coordinates (coordinate -> [i, j]) of growth candidates

    """
    candidates = []
    for i in range(N):
        for j in range(N):

            # check all possible growth candidates if on object
            if objects[i, j] == 1:

                # directions west, east, sout, north and apply periodic boundary conditions
                for dxdy in np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]):
                    i_cand = (dxdy[0] + i) % N
                    j_cand = dxdy[1] + j

                    # check if neighbour doesn't lay outside boundaries
                    if j_cand >= 0 and j_cand < N-1:

                        # check if c != 0, and thus if there is a growth probability
                        if c[i_cand, j_cand] > 0:
                            candidates.append([i_cand, j_cand])

    return np.array(candidates)


@numba.njit()
def get_growth_probabilities(c, candidates, eta):
    """Calculate growth probability for each candidate according to growth probabilities formula

    Args:
        c (2D nparray): concentration at each coordinate
        candidates (2D nparray): array of coordinates (coordinate -> [i, j]) of growth candidates
        eta (float (0, 2)): parameter to change form of object

    Returns:
        probabilities (1D nparray): growth probabilities of each corresponding coordinate

    """
    if eta == 0:
        probabilities = np.array([1/len(candidates) for candidate in candidates])
    else:
        total = np.sum(np.array([c[i, j]**eta for i, j in candidates]))
        probabilities = np.array([c[i, j]**eta/total for i, j in candidates])

    return probabilities


@numba.njit()
def sor(c, objects, omega=1.7):
    """Successive Over Relaxation (SOR)

    Args:
        c (2D nparray): concentration at each coordinate
        objects (2D nparray): array of whole grid space. Value of 1 means object, 0 means no object
        omega (float (1, 2)): relaxation parameter for SOR method

    Returns:
        c (2D nparray): concentration at each coordinate

    """
    c_new = np.copy(c)
    delta = np.inf
    eps = 1e-5
    n_iters = 0
    while delta > eps:
        n_iters += 1
        for i in range(len(c)):
            for j in range(1, len(c[0])-1):
                if objects[i, j] == 1:
                    continue
                c_new[i, j] = (omega/4) * (c_new[(i+1)%len(c), j] + c_new[i-1, j] \
                + c_new[i, j+1] + c_new[i, j-1]) + (1 - omega) * c_new[i, j]

                # to ensure that c never gets negative, probably caused by numerical errors
                if c_new[i, j] < 0:
                    c_new[i, j] = 0

        delta = np.max(np.abs(c_new - c))
        c = np.copy(c_new)

    return c_new, n_iters


def run_dla_prob_model(N, eta, omega=1.8):
    """DLA model using growth probability. Simulation continues until one object reaches to
    the y-limit (N - 1)

    Args:
        N (int): lattice size
        eta (float (0, 2)): parameter to change form of object

    """
    print("DLA probability model with eta, N:", eta, N)

    # start with analytical solution for domain, with at y = 1, c = 1
    c = np.array([[j/(N-1) for j in range(N)] for i in range(N)])

    # start with a single object (object has c = 0)
    objects = np.zeros((N, N))
    objects[N//2][0] = 1
    c[N//2][0] = 0

    n_iters = 0
    n_iters_sor = 0

    # perform simulation
    while True:
        n_iters += 1

        # perform sor iteration to determine new nutrient field
        c, n_temp = sor(c, objects, omega)
        n_iters_sor += n_temp

        # determine growth candidates
        candidates = get_growth_candidates(N, c, objects)

        # if no candidates, stop program, growth limit reached
        if len(candidates) == 0:
            print("Growth limit reached, no candidates left.")
            break

        # calculate growth probabilities
        probabilities = get_growth_probabilities(c, candidates, eta)

        # check if probabilities are existing
        if len(probabilities) == 0:
            print("Growth limit reached, no non-zero concentration growth candidates left.")
            break

        # choose growth location and add location to objects
        index = np.random.choice(len(candidates), p=probabilities)

        objects[candidates[index][0], candidates[index][1]] = 1
        c[candidates[index][0], candidates[index][1]] = 0       # set c=0 on object

        # check if maximum height is reached
        if candidates[index][1] == N - 2:
            break

    return c, objects, n_iters_sor / n_iters


def dla_prob_model():

    plt.rcParams.update({"font.size": 14})

    N = 100
    eta = 1
    omega = 1.8

    # perform model ones, to get the functions compiled, otherwise the timing is off for the
    # first iteration of calculating the simulation time
    run_dla_prob_model(10, 1)

    c, objects, *_ = run_dla_prob_model(N, eta)

    plt.matshow(objects)
    plt.title("DLA object $N={}$, $\eta = {}$\n".format(N, eta))
    plt.colorbar()
    plt.savefig("results/DLA_object_N{}_eta{}.png".format(N, eta))

    plt.matshow(c)
    plt.title("DLA diffusion plot $N={}$, $\eta = {}$\n".format(N, eta))
    plt.colorbar()
    plt.savefig("results/DLA_diffusion_N{}_eta{}.png".format(N, eta))

    # create nice DLA plot of 4 different eta values
    fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
    plt.rcParams.update({"font.size": 14})
    im = None
    etas = [0.25, 0.75, 1.25, 1.75]
    for i, ax in enumerate(axs.flatten()):
        im = ax.matshow(run_dla_prob_model(N, etas[i])[1].T, origin="lower", extent=[0., 1., 0., 1.])
        ax.set_title("DLA $\eta={}$".format(etas[i]))

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
    plt.savefig("results/DLA_prob/DLA_vary_eta.png")
    plt.show()

    time_start = time.time()

    # simulation time as a function of lattice size N
    n_reps = 5
    Ns = np.linspace(10, 150, 15, dtype=int)
    sim_times = np.zeros((len(Ns), n_reps))
    print("Time:{}".format(15*10*6/60))
    for i, N in enumerate(Ns):
        for rep in range(n_reps):
            time_start_N = time.time()
            run_dla_prob_model(N, eta, omega)
            sim_times[i, rep] = time.time() - time_start_N

    # calculate statistics
    sim_times_means = [np.mean(i) for i in sim_times]
    sim_times_conf_int = [(np.std(i, ddof=1) * 1.96) / np.sqrt(n_reps) for i in sim_times]

    plt.figure()
    plt.rcParams.update({"font.size": 14})
    plt.title("DLA simulation time over the lattice size $\eta={}$".format(eta))
    plt.errorbar(Ns, sim_times_means, yerr=sim_times_conf_int, color="orange", fmt="none", zorder=0)
    plt.scatter(Ns, sim_times_means, color="blue", s=15)
    plt.xlabel("Lattice size $N$")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig("results/DLA_prob/DLA_time_N_eta{}.png".format(eta))

    # number of iterations as a function of omega and lattice size
    n_reps = 5
    Ns = [10, 50, 100, 150]
    omegas = np.linspace(1, 1.9, 20, dtype=float)

    plt.figure()
    plt.rcParams.update({"font.size": 14})
    plt.title("DLA simulation iterations over $\omega$, $\eta={}$".format(eta))
    fmts = ["^", "x", "v", "s", "+", "d"]
    for j, N in enumerate(Ns):
        sim_n_iters_omega = np.zeros((len(omegas), n_reps))


        for i, omega in enumerate(omegas):
            for rep in range(n_reps):
                *_, n_iters_omega = run_dla_prob_model(N, eta, omega)
                sim_n_iters_omega[i, rep] = n_iters_omega

        # calculate statistics
        sim_n_iters_omega_means = [np.mean(i) for i in sim_n_iters_omega]
        sim_n_iters_omega_conf_int = [(np.std(i, ddof=1) * 1.96) / np.sqrt(n_reps) for i in sim_n_iters_omega]


        plt.errorbar(omegas, sim_n_iters_omega_means, yerr=sim_n_iters_omega_conf_int, capsize=3,
                    fmt=fmts[j], zorder=0, label="$N={}$".format(N))

    plt.xlabel("$\omega$")
    plt.ylabel("Number of iterations SOR method")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/DLA_prob/DLA_vary_omega_vary_N_eta{}.png".format(eta))

    print("DLA probability model simulation time: {:.2f} seconds.\n".format((time.time() - time_start)))

    plt.show()

    return


if __name__ == '__main__':
    dla_prob_model()
