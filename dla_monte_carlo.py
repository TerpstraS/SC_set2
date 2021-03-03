import numpy as np
import numba
import matplotlib.pyplot as plt
import time

@numba.njit
def get_new_walker(grid):
    while True:
        walker_x = np.random.randint(0, len(grid))

        # y should be 99 here! Not a random number
        walker_y = len(grid)-1
        # print(walker_x, walker_y)
        if grid[walker_x, walker_y] == 1:
            continue
        else:
            return walker_x, walker_y


@numba.njit
def walk(walker_x, walker_y, N, grid):
    direction = np.random.randint(0, 4)
    if direction == 0:
        walker_y+=1
    if direction == 1:
        walker_y-=1
    if direction == 2:
        walker_x+=1
    if direction == 3:
        walker_x-=1

    if walker_y >= N or walker_y < 0:
        walker_x, walker_y = get_new_walker(grid)
        return walk(walker_x, walker_y, N, grid)
    if walker_x >= N:
        walker_x = 0
    if walker_x < 0:
        walker_x = N-1

    return walker_x, walker_y

@numba.njit
def next_to_structure(walker_x, walker_y, grid):
    if walker_y < len(grid) - 1:
        if grid[walker_x, walker_y + 1] == 1:
            return True
    if walker_y > 0:
        if grid[walker_x, walker_y - 1] == 1:
            return True
    if walker_x < len(grid) - 1:
        if grid[walker_x + 1, walker_y] == 1:
            return True
    if walker_x > 0:
        if grid[walker_x - 1, walker_y] == 1:
            return True
    return False

@numba.njit
def dla_monte_carlo(N, sticking_prob = 1):
    """DLA model using Monte Carlo simulation
    Args:
        N (int): lattice size
        num_walkers(int): number of walkers that will be used
    """
    # print("DLA Monte Carlo model with N = {}.".format(N))
    # print("")

    grid = np.zeros((N, N))
    struct_x = int(N/2)
    struct_y = 0
    grid[struct_x, struct_y] = 1

    while True:
        walker_x, walker_y = get_new_walker(grid)
        stuck = False

        while not stuck:
            if next_to_structure(walker_x, walker_y, grid):
                if np.random.uniform(0, 1) < sticking_prob:
                    stuck = True
                    grid[walker_x, walker_y] = 1
                    if walker_y == N-2:
                        return grid

                # Added this else to fix walkers not getting out of this if statement,
                # and thus sticking probability not working
                else:
                    walker_x, walker_y = walk(walker_x, walker_y, N, grid)

            else:
                walker_x, walker_y = walk(walker_x, walker_y, N, grid)

    return grid

def run_dla_monte_carlo_experiments(N, sticking_prob=1):
    print("in main")

    plt.rcParams.update({"font.size": 14})

    time_start = time.time()
    grid = dla_monte_carlo(100, 1)
    print("DLA Monte Carlo model simulation time: {:.2f} seconds.\n".format(time.time() - time_start))

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.matshow(grid.T, origin="lower", extent=[0., 1., 0., 1.])
    ax.xaxis.tick_bottom()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.yaxis.tick_left()
    plt.savefig("DLA monte carlo 1")
    plt.show()

    fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
    plt.rcParams.update({"font.size": 14})
    im = None
    probabilities = [0.01, 0.05, 0.1, 0.5]
    for i, ax in enumerate(axs.flatten()):
        im = ax.matshow(dla_monte_carlo(N, probabilities[i]).T, origin="lower", extent=[0., 1., 0., 1.])
        ax.set_title("DLA $P_s={}$".format(probabilities[i]))

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
    plt.savefig("DLA monte carlo Ps")
    plt.show()

    sizes = []
    ps = np.linspace(0.005, 0.5, 50)
    for p in ps:
        # we will use this array to store temporary results
        temp = []
        for i in range(5):
            grid = dla_monte_carlo(N, p)
            size = np.sum(grid)
            temp.append(size)
        sizes.append(np.mean(temp))
    fig = plt.figure()
    plt.scatter(ps, sizes)
    plt.xlabel('$P_s$')
    plt.ylabel('Structure size')

if __name__ == "__main__":
    run_dla_monte_carlo_experiments(100)

