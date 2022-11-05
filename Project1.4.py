import argparse
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)


def discrete_decay(initial_nuclei, decay_rate):
    nuclei_time = [initial_nuclei]
    nuclei_left = initial_nuclei
    while nuclei_left > 0:
        nuclei_decay = sum(1 for num in rng.random(nuclei_left) - decay_rate if num < 0)
        nuclei_left -= nuclei_decay
        nuclei_time.append(nuclei_left)
    return nuclei_time


def main():
    decay_rate = 0.03
    if args.part_b:
        decay_rate = 0.3
    nuclei_sample = [10**(i+1) for i in range(5)]
    for initial_nuclei in nuclei_sample:
        nuclei_time = discrete_decay(initial_nuclei, decay_rate)
        t = range(np.size(nuclei_time))
        continuous_decay = initial_nuclei * np.exp(-decay_rate * np.array(t))
        plt.plot(t, np.log(nuclei_time), label='$N_{0} = $' + f'{initial_nuclei:,.0f}')
        plt.plot(t, np.log(continuous_decay))
    plt.title(f'Radioactive decay for a decay rate of $ \\lambda = {decay_rate}$ per second')
    plt.xlabel('Time (in seconds)')
    plt.ylabel('$ \\log{N(t)}$')
    plt.legend()
    if not args.save:
        plt.show()
    elif args.part_b:
        plt.savefig('P1.4b.png', dpi=1200)
    else:
        plt.savefig('P1.4a.png', dpi=1200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_b', default=False, action="store_true", help="Obtain results for part b)")
    parser.add_argument('--save', default=False, action="store_true", help="Save plots instead of showing them")
    args = parser.parse_args()
    main()
