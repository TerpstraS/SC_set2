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


@numba.njit
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


@numba.njit
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
    while delta > eps:
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

    return c_new


def dla_prob_model(N, eta):
    """DLA model using growth probability. Simulation continues until one object reaches to
    the y-limit (N - 1)

    Args:
        N (int): lattice size
        eta (float (0, 2)): parameter to change form of object

    """
    print("DLA probability model with eta = {}, N = {}.".format(eta, N))

    # start with analytical solution for domain, with at y = 1, c = 1
    c = np.array([[j/(N-1) for j in range(N)] for i in range(N)])

    # start with a single object (object has c = 0)
    objects = np.zeros((N, N))
    objects[N//2][0] = 1
    c[N//2][0] = 0

    time_start = time.time()

    # perform simulation
    while True:

        # perform sor iteration to determine new nutrient field
        c = sor(c, objects, omega=1.8)

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
        index = np.random.choice(np.linspace(0, len(candidates)-1, len(candidates), dtype=int),
            p=probabilities)

        objects[candidates[index][0], candidates[index][1]] = 1
        c[candidates[index][0], candidates[index][1]] = 0       # set c=0 on object

        # check if maximum height is reached
        if candidates[index][1] == N - 2:
            print("Reached maximum growth height.")
            break

    print("DLA probability model simulation time: {:.2f} seconds.\n".format(time.time() - time_start))

    plt.matshow(objects)
    plt.title("DLA object $\eta = {}$\n".format(eta))
    plt.colorbar()

    plt.matshow(c)
    plt.title("DLA diffusion plot $\eta = {}$\n".format(eta))
    plt.colorbar()
    plt.show()

    return
