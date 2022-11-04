import argparse
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

a = -1
b = 1


def random_walk(n):
    values_x = (b - a) * rng.random(n) + a
    values_y = (b - a) * rng.random(n) + a
    length = np.sqrt(values_x ** 2 + values_y ** 2)
    steps_x = values_x / length
    steps_y = values_y / length
    x = np.add.accumulate(steps_x)
    y = np.add.accumulate(steps_y)
    x = np.concatenate(([0], x))
    y = np.concatenate(([0], y))
    return x, y


def get_histogram(data, n, m):
    plt.hist(data, bins=100, ec='black', lw=0.5, color='dodgerblue', label='Data generated')
    plt.title(f'Distance from the origin after {n:,.0f} steps \n for {m:,.0f} independent simulations')
    plt.xlabel('Distance $ R_{N}$')
    plt.ylabel('Frequency')
    plt.savefig('P1-2b1.png', dpi=1200)
    return


def get_rms_distance(x, y):
    dx = np.array([x[i+1] - x[i] for i in range(np.size(x) - 1)])
    dy = np.array([y[i+1] - y[i] for i in range(np.size(y) - 1)])
    rms_distance = np.sqrt(np.sum(dx**2 + dy**2))
    return rms_distance


def main():
    if args.part_a:
        n = 1000  # Number of steps
        x1, y1 = random_walk(n)
        x2, y2 = random_walk(n)
        x3, y3 = random_walk(n)
        plt.plot(x1, y1, 'b', x2, y2, 'g', x3, y3, 'r')
        plt.axhline(0, color='k')
        plt.axvline(0, color='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'3 random walks in 2 dimensions after {n:,.0f} steps')
        plt.savefig('Proj1.2a.png', dpi=1200)
    else:
        n = 10000  # Number of steps
        m = 1000  # Number of independent simulations
        last_steps = []
        for i in range(m):
            x, y = random_walk(n)
            final_distance = np.sqrt(x[-1]**2 + y[-1]**2)
            last_steps.append(final_distance)
        get_histogram(last_steps, n, m)  # Save an image containing the histogram.
        # Comparison of the computed root-mean-square to the theoretical value for different number of steps
        rms_distances = []
        steps_list = list(range(1000, 20000, 1000))
        for steps in steps_list:
            x, y = random_walk(steps)
            rms = get_rms_distance(x, y)
            rms_distances.append(rms)
        plt.clf()  # Needed to clear the contains of plt.plot(), i.e., so that the histogram doesn't appear again
        plt.plot(steps_list, rms_distances, 'bo', markersize=6, label='$ \\sqrt{\\langle R^{2} \\rangle_{N}} $')
        steps_min, steps_max = plt.xlim()
        steps = np.linspace(steps_min, steps_max, 100)
        plt.plot(steps, np.sqrt(steps), 'r', label='$ \\sqrt{N} \\cdot r_{rms} $')
        plt.xlabel('Number of steps ($ N $)')
        plt.ylabel('Root-mean-square distance')
        plt.title('Comparison between the computed root-mean-square distance \n '
                  '$ R_{rms, N} = \\sqrt{\\langle R^{2} \\rangle_{N}} $ and the theoretical curve '
                  '$ R_{rms} = \\sqrt{N} \\cdot r_{rms}$')
        plt.legend()
        plt.savefig('P1-2b2.png', dpi=1200)
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_a', default=False, action="store_true", help="Obtain results for part a)")
    args = parser.parse_args()
    main()
