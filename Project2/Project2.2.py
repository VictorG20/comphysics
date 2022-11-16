import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

rng = np.random.default_rng(42)

'''
TO DO:
-Optimize

https://itp.uni-frankfurt.de/~mwagner/teaching/C_WS19/projects/Ising_proj.pdf
- What I'm understanding is that:
    1. One 'full update' corresponds to making L^2 trials.
    2. N_{skip} is the amount of updates we do before we actually 'take a look' at
       the state again. Therefore, we do N_{skip} * L^2 trials before we take any observable
       like the energy or the magnetization. 
    3. N is the number of generated configurations to be considered, i.e., the number of
       states on which we will calculate averages and such.
    4. Do not restart the state of the system after every full update but instead
       use the last state and perform one full update on it.
'''


def get_acceptance_probabilities(beta):
    beta = float(beta)
    energy_diff = sorted(set(2. * spin_mid * (spin_left + spin_right + spin_above + spin_below)
                             for spin_mid in [-1, 1] for spin_left in [-1, 1] for spin_right in [-1, 1]
                             for spin_above in [-1, 1] for spin_below in [-1, 1]))
    pos_energy_diff = [delta_E for delta_E in energy_diff if delta_E > 0]
    acceptance_prob = [np.exp(-delta_E * beta) for delta_E in pos_energy_diff]
    return pos_energy_diff, acceptance_prob


def mc_step(state, pos_energy_diff, acceptance_prob):
    trial_files = rng.integers(length, size=skip*spins)
    trial_columns = rng.integers(length, size=skip*spins)
    for i, j in zip(trial_files, trial_columns):
        energy_difference = 2 * state[i, j] * (state[i, (j + 1) % length] + state[i, j - 1]
                                               + state[(i + 1) % length, j] + state[i - 1, j])
        if energy_difference > 0 and rng.random(1) > acceptance_prob[pos_energy_diff.index(energy_difference)]:
            continue
        state[i, j] *= -1
    return


def main():
    beta_values = [float(i)/20 for i in range(21)]
    mean_magnetisation = []
    for beta in beta_values:
        print("beta =", beta)
        state = -np.ones((length, length), dtype=int)
        pos_energy_diff, acceptance_prob = get_acceptance_probabilities(beta)
        magnetisation_values = [np.sum(state)/spins]
        start = time.time()
        for i in range(configurations):
            mc_step(state, pos_energy_diff, acceptance_prob)
            magnetisation_values.append(np.sum(state)/spins)
            if i % (configurations//10) == 0:
                print(i//(configurations//100))
                end = time.time()
                print(end-start)
        mean_magnetisation.append(np.abs(np.mean(magnetisation_values)))
    plt.plot(beta_values, mean_magnetisation, 'o')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default=False, action="store_true", help="Save plots generated instead of showing.")
    args = parser.parse_args()
    length = 30
    spins = length ** 2
    configurations = 500
    skip = 10
    main()
