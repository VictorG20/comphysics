import argparse
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

'''
In terms of energy, the sum over nearest neighbors has only the possible counting values of:
uuu, ddd -> E = 2 J
uud, duu, udd, ddu -> E = 0
udu, dud -> E = -2 J

This implies that, when flipping the spin in the middle to get a new configuration, the possible energy differences
are only Delta_E = 4, 0 and -4 (in units of J)

TO DO:
-For part a). Implement state visualization until there has been some change?
'''


def visualize_states(save_trials, saved_states, spins):
    colors = ['r', 'r', 'b']
    plt.axis([0.5, 20.5, -1, 19])
    ax = plt.gca()
    ax.set_xticks(np.arange(1, 21, 1.0), labels='')
    ax.set_yticks(np.arange(0, 20, 2.0), labels=save_trials)
    ax.set_xlabel('Spins')
    ax.set_ylabel('Number of trial')
    ax.set_aspect(0.9)
    down_spins = [f'{state.count(-1)}/20' for state in saved_states]

    counter = 0
    for state in saved_states:
        for i in range(spins):
            c = plt.Circle((i + 1, 2 * counter), radius=0.3, color=colors[state[i]])
            plt.gca().add_artist(c)
        counter += 1

    # Create a second y-axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(0, 20, 2), labels=down_spins)
    ax2.set_ylabel('Amount of spins down')
    ax2.set_aspect(0.9)
    plt.tight_layout()

    if args.save:
        plt.savefig('P2-1a.png', dpi=1200)
    else:
        plt.title('System at the given trials')
        plt.show()
    return


def trial_spin_flips(spins, state, energy, thermal_energy, flips):
    energy_differences = [4., 0., -4.]  # In units of 'J'.
    acceptance_prob = [min(np.exp(-delta_E / thermal_energy), 1.) for delta_E in energy_differences]

    energies = [energy]  # Keep track of the energy at each trial.

    # Only used for part (a).
    save_trials = [0, 5, 100, 145, 200, 300, 350, 400, 450, 500]  # Chosen times to visualize the system in the plot.
    saved_states = [state.copy()]  # Save states to be plotted later, including the initial state.
    # ---------------------------------------------------------------------------------------------------------

    trials = rng.integers(low=0, high=spins, size=flips)  # Chosen spin sites to try to flip. 'high' is exclusive.

    for i in range(flips):
        energy_difference = 2. * state[trials[i]] * \
                            (state[(trials[i] + 1) % spins] + state[trials[i] - 1])  # Modulo due to Periodic B.C.
        if energy_difference > 0 and rng.random(1) > acceptance_prob[0]:
            energies.append(energy)
            if args.part == 'a' and i + 1 in save_trials:
                saved_states.append(state.copy())
            continue
        state[trials[i]] = -1 * state[trials[i]]  # Flip spin
        energy += energy_difference  # Update energy of the system
        energies.append(energy)
        if args.part == 'a' and i + 1 in save_trials:
            saved_states.append(state.copy())

    if args.part == 'a':
        return save_trials, saved_states
    elif args.part == 'b':
        return energies


def main():
    spins = 20  # Number of spins in the system
    state = [-1 for _ in range(spins)]  # All spins initially pointing in the same direction. Here it is downwards.
    energy = -spins  # Initial energy is known for a cold initial state. Take the appropriate sign.

    if args.part == 'a':
        thermal_energy = 1.
        flips = 500  # Individual trial spin flips
        save_trials, saved_states = trial_spin_flips(spins, state, energy, thermal_energy, flips)
        visualize_states(save_trials, saved_states, spins)
    elif args.part == 'b':
        thermal_energies = [0.1, 1, 10]
        flips = 1000
        colors = ['darkviolet', 'orange', 'red']
        for i in range(np.size(thermal_energies)):
            energies = trial_spin_flips(spins, state, energy, thermal_energies[i], flips)
            t = np.linspace(0, flips, num=np.size(energies), endpoint=True)
            plt.plot(t, energies, label='$k_{B}T =$' + f'{thermal_energies[i]}', color=colors[i])
        plt.xlabel('Time (in a single trial spin flip)')
        plt.ylabel('Energy (in units of $J$)')
        plt.legend()
        if args.save:
            plt.savefig('P2-1b1.png', dpi=1200)
        else:
            plt.title(f'Time evolution of the energy for a single simulation \n under {flips:,.0f} spin flip trials')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default=False, action="store_true", help="Save plots generated instead of showing.")
    parser.add_argument('--part', type=str, default='a', help="Choose the code for the given part to be executed.")
    args = parser.parse_args()
    main()
