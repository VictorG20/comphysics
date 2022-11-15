import argparse
import numpy as np

rng = np.random.default_rng(42)

'''
TO DO:
-Implement equilibrium criteria and add it to 'spin_flip_trials' function.
'''


def spin_flip_trials(state, thermal_energy, flips, equilibrate=False):
    energy = -1. * sum([state[i][j]*(state[i][j-1] + state[i][(j+1) % args.length] + state[i-1][j] +
                                     state[(i+1) % args.length][j])
                        for i in range(args.length) for j in range(args.length)])
    energies = [energy]
    magnetisation = np.sum(state)
    magnetisation_values = [magnetisation]

    energy_differences = sorted(set(2. * spin_mid * (spin_left + spin_right + spin_above + spin_below)
                                    for spin_mid in [-1, 1] for spin_left in [-1, 1] for spin_right in [-1, 1]
                                    for spin_above in [-1, 1] for spin_below in [-1, 1]))
    pos_energy_differences = [delta_E for delta_E in energy_differences if delta_E > 0]
    acceptance_prob = [np.exp(-delta_E / thermal_energy) for delta_E in pos_energy_differences]

    if equilibrate:
        print('Under construction')
    #     touched_spins = []
    #     while energy_deviation < args.length**2:

    trial_files = rng.integers(args.length, size=flips)
    trial_columns = rng.integers(args.length, size=flips)
    for i, j in zip(trial_files, trial_columns):
        energy_difference = 2. * state[i][j] * (state[i][(j + 1) % args.length] + state[i][(j - 1) % args.length]
                                                + state[(i + 1) % args.length][j] + state[(i - 1) % args.length][j])
        if energy_difference > 0 and rng.random(1) > acceptance_prob[pos_energy_differences.index(energy_difference)]:
            energies.append(energy)
            magnetisation_values.append(magnetisation)
            continue
        state[i][j] = -1 * state[i][j]  # Flip spin
        energy += energy_difference  # Update energy of the system
        magnetisation += 2. * state[i][j]  # Update magnetisation of the system (with new value of the spin flipped)
        energies.append(energy)
        magnetisation_values.append(magnetisation)
    return energies, magnetisation_values


def main():
    state = [[-1 for _ in range(args.length)] for _ in range(args.length)]
    energies, magnetization_values = spin_flip_trials(state, thermal_energy=1., flips=10000, equilibrate=True)
    print(magnetization_values[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=30, help="Number of spins per side in a square lattice.")
    parser.add_argument('--save', default=False, action="store_true", help="Save plots generated instead of showing.")
    args = parser.parse_args()
    main()
