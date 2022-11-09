import argparse
import numpy as np

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
    flips = 100  # Individual trial spin flips
    state = [-1 for _ in range(spins)]  # All spins pointing in the same direction. Here they all point downwards.
    energy = spins * state[0]  # Initial energy is known for a cold initial state. Take the appropriate sign.

    energy_differences = [4., 0., -4.]  # In units of 'J'.
    acceptance_prob = [min(np.exp(-delta_E/thermal_energy), 1.) for delta_E in energy_differences]
    counter = 0

    trials = rng.integers(low=0, high=spins, size=flips)  # Chosen spin sites to try to flip. 'high' is exclusive.
    for trial in trials:
        counter += 1
        energy_difference = -2.*state[trial]*(state[(trial+1) % spins] + state[trial-1])  # Modulo due to Periodic B.C.
        if energy_difference > 0 and rng.random(1) > acceptance_prob[0]:
            # if counter % 5 == 0:
            #     print(counter, state)
            continue
        state[trial] = -1 * state[trial]
        energy += energy_difference
        # if counter % 5 == 0:
        #     print(counter, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
