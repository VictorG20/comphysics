import argparse
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.collections as mc

rng = np.random.default_rng(42)


'''
In terms of energy, the sum over nearest neighbors has only the possible counting values of:
uuu, ddd -> E = 2 J
uud, duu, udd, ddu -> E = 0
udu, dud -> E = -2 J

This implies that, when flipping the spin in the middle to get a new configuration, the possible energy differences
are only Delta_E = 4, 0 and -4 (in units of J)
'''


def main():
    spins = 20  # Number of spins in the system
    thermal_energy = 1.
    flips = 500  # Individual trial spin flips
    state = [-1 for _ in range(spins)]  # All spins pointing in the same direction. Here they all point downwards.
    energy = spins * state[0]  # Initial energy is known for a cold initial state. Take the appropriate sign.

    energy_differences = [4., 0., -4.]  # In units of 'J'.
    acceptance_prob = [min(np.exp(-delta_E/thermal_energy), 1.) for delta_E in energy_differences]
    counter = 0

    # Plotting
    colors = ['r', 'r', 'b']
    visualize = [0, 2, 5, 10, 20, 50, 100, 200, 300, 500]
    plt.axis([0, 21, -1, 19])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(np.arange(1, 21, 1.0), labels='')
    plt.yticks(np.arange(0, 20, 2), labels=visualize)

    for i in range(spins):
        c = plt.Circle((i+1, 0), radius=0.3, color=colors[state[i]])
        plt.gca().add_artist(c)
    order = 1

    trials = rng.integers(low=0, high=spins, size=flips)  # Chosen spin sites to try to flip. 'high' is exclusive.
    for trial in trials:
        counter += 1
        if counter in visualize:
            order += 1
            for i in range(spins):
                c = plt.Circle((i + 1, 2*(order - 1)), radius=0.3, color=colors[state[i]])
                plt.gca().add_artist(c)
        energy_difference = -2.*state[trial]*(state[(trial+1) % spins] + state[trial-1])  # Modulo due to Periodic B.C.
        if energy_difference > 0 and rng.random(1) > acceptance_prob[0]:
            continue
        state[trial] = -1 * state[trial]
        energy += energy_difference
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
