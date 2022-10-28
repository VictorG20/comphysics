import argparse
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)


def main():
    a = -1
    b = 1
    n = 1000  # Number of steps
    values_x = (b - a)*rng.random(n) + a
    values_y = (b - a)*rng.random(n) + a
    length = np.sqrt(values_x**2 + values_y**2)
    steps_x = values_x/length
    steps_y = values_y/length
    x = np.add.accumulate(steps_x)
    y = np.add.accumulate(steps_y)
    x = np.concatenate(([0], x))
    y = np.concatenate(([0], y))
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Random walk in 2 dimensions for {n} steps')
    plt.show()


if __name__ == '__main__':
    main()
