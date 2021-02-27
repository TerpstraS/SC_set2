import numpy as np
import numba
import matplotlib.pyplot as plt


def get_new_walker(grid):
    while True:
        walker_x, walker_y = np.random.randint(0, len(grid), 2)
        if grid[walker_x, walker_y] == 1:
            continue
        else:
            return walker_x, walker_y

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

def dla_monte_carlo(N, num_walkers = 5, sticking_prob = 1):
    """DLA model using Monte Carlo simulation
    Args:
        N (int): lattice size
        num_walkers(int): number of walkers that will be used
    """
    print("DLA Monte Carlo model with N = {}.".format(N))
    print("")

    grid = np.zeros((N, N))
    struct_x = int(N/2)
    struct_y = 0
    grid[struct_x, struct_y] = 1

    for _ in range(num_walkers):
        walker_x, walker_y = get_new_walker(grid)
        stuck = False

        while not stuck:
            if next_to_structure(walker_x, walker_y, grid):
                if np.random.uniform(0, 1) < sticking_prob:
                    stuck = True
                    grid[walker_x, walker_y] = 1
            else:
              walker_x, walker_y = walk(walker_x, walker_y, N, grid)
    return grid

if __name__ == "__main__":
    print("in main")
    grid = dla_monte_carlo(50, 100)

    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(grid)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
