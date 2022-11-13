import argparse
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)


'''
In terms of energy, the sum over nearest neighbors has only the possible counting values of:
uuu, ddd -> E = -2 J
uud, duu, udd, ddu -> E = 0
udu, dud -> E =  2 J

This implies that, when flipping the spin in the middle to get a new configuration, the possible energy differences
are only Delta_E = 4, 0 and -4 (in units of J)

TO DO:
-For part a). Implement state visualization until there has been some change?
-Would it be convenient to carry the plotting in a separate function? Maybe not because titles are always different
 and so are the labels of the axes. Maybe pass them as strings into the function together with the data?
-Can the function 'trial_spin_flips' be standardized better to avoid different returns?
-For part b). Maybe create also a separate plot with the Monte-Carlo error estimate.
-Both thermal energies and energies are in units of 'J'
-Fix titles of plots. Some of them read weird.

Observations:
-Remember that lists are mutable objects. Therefore, if you change them inside a function, they are also changed
 outside of it and THEY MUST BE RESTORED before running a simulation again. Otherwise you're starting a simulation
 with the last state of the system from the previous simulation.
'''


def visualize_states(save_trials, saved_states):
    spins = np.shape(saved_states)[1]
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
        plt.savefig(f'P2-1{args.part}.png', dpi=1200)
    else:
        plt.title('System at the given trials')
        plt.show()
    return


def trial_spin_flips(state, thermal_energy, flips):
    spins = np.size(state)
    energy = -1. * np.sum([state[i]*state[(i+1) % spins] for i in range(spins)])

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
    else:
        return energies


def part_b(spins):
    thermal_energies = [0.1, 1, 10]
    flips = 1000
    colors = ['darkviolet', 'orange', 'red']
    for i in range(np.size(thermal_energies)):
        state = [-1 for _ in range(spins)]  # All spins initially pointing in the same direction (here downwards).
        energies = trial_spin_flips(state, thermal_energies[i], flips)
        t = np.linspace(0, flips, num=np.size(energies), endpoint=True)
        plt.plot(t, energies, label='$k_{B}T =$' + f'{thermal_energies[i]}', color=colors[i])
    plt.xlabel('Time (in single trials for spin flip)')
    plt.ylabel('Energy (in units of $J$)')
    plt.legend()
    if args.save:
        plt.savefig('P2-1b1.png', dpi=1200)
    else:
        plt.title(f'Time evolution of the energy for a single simulation \n under {flips:,.0f} spin flip trials')
        plt.show()

    plt.clf()  # Clear plot variable
    simulations = 100

    for i in range(np.size(thermal_energies)):
        simulation_energy = []
        for j in range(simulations):
            state = [-1 for _ in range(spins)]
            energies = trial_spin_flips(state, thermal_energies[i], flips)
            simulation_energy.append(energies)
        t = np.linspace(0, flips, num=np.shape(simulation_energy)[1], endpoint=True)
        averaged_energies = np.mean(simulation_energy, axis=0)
        squared_energies = [[energy ** 2 for energy in energies] for energies in simulation_energy]
        averaged_squared_energies = np.mean(squared_energies, axis=0)
        monte_carlo_error = np.sqrt((averaged_squared_energies - averaged_energies ** 2) / spins)
        plt.plot(t, averaged_energies, label='$k_{B}T =$' + f'{thermal_energies[i]}', color=colors[i])
        plt.fill_between(t, averaged_energies - monte_carlo_error, averaged_energies + monte_carlo_error,
                         color=colors[i], alpha=0.3)
    plt.xlabel('Time (in single trials for spin flip)')
    plt.ylabel('Mean energy $\\langle E \\rangle$')
    plt.legend()
    if args.save:
        plt.savefig('P2-1b2.png', dpi=1200)
    else:
        plt.title(f'Time evolution of the mean energy over \n'
                  f'{simulations:,.0f} simulations for {flips:,.0f} spin flip trials')
        plt.show()
    return


def main():
    spins = 20  # Number of spins in the system

    if args.part == 'a':
        state = [-1 for _ in range(spins)]  # All spins initially pointing in the same direction. Here it is downwards.
        thermal_energy = 1.
        save_trials, saved_states = trial_spin_flips(state, thermal_energy, flips=500)
        visualize_states(save_trials, saved_states)
    elif args.part == 'b':
        part_b(spins)
    elif args.part == 'c' or 'd':
        thermal_energies = [i+1 for i in range(10)]  # Choose kT = 1, 2, ..., 10
        simulations = 100  # Number of independent simulations for each thermal energy
        avg_energy_particle = []
        specific_heat_particle = []
        for thermal_energy in thermal_energies:
            time_avg_energies = []
            specific_heats = []
            for j in range(simulations):
                state = [-1 for _ in range(spins)]
                trial_spin_flips(state, thermal_energy, flips=1000)  # Run the initial state for 1000 flips first.
                energies = trial_spin_flips(state, thermal_energy, flips=1000)  # Now get the energies after eq.
                average_energy = np.mean(energies)
                average_squared_energy = np.mean(np.array(energies) ** 2)
                specific_heat = (average_squared_energy - average_energy**2)/(thermal_energy**2)
                time_avg_energies.append(average_energy)
                specific_heats.append(specific_heat)
            avg_energy_particle.append(np.mean(time_avg_energies)/spins)
            specific_heat_particle.append(np.mean(specific_heats)/spins)

        temperatures = np.linspace(np.min(thermal_energies), np.max(thermal_energies), num=100)
        if args.part == 'c':
            plt.plot(thermal_energies, avg_energy_particle, 'o', color='b', label='Simulated values')
            plt.plot(temperatures, -np.tanh(1/temperatures), color='r',
                     label='$-J \\cdot \\tanh{\\frac{J}{k_{B}T}}$')
            ax = plt.gca()
            ax.set_xticks(np.arange(1, 11, 1.0))
            plt.xlabel('$k_{B}T$ (in units of $J$)')
            plt.ylabel('$\\frac{1}{N} \\langle E \\rangle_{t}$ (in units of $J$)')
            plt.legend()
            if args.save:
                plt.savefig(f'P2-1{args.part}.png', dpi=1200)
            else:
                plt.title('Mean energy per particle over 1,000 trial spin flips, after \n '
                          f'equilibrium has been reached, for {simulations:,.0f} simulations each')
                plt.show()
        else:
            plt.plot(thermal_energies, specific_heat_particle, 'o', color='b', label='Simulated values')
            plt.plot(temperatures, 1/((temperatures**2) * (np.cosh(1/temperatures))**2), color='r',
                     label='$(J/k_{B}T)^{2}/\\cosh^{2}{(J/k_{B}T)}$')
            ax = plt.gca()
            ax.set_xticks(np.arange(1, 11, 1.0))
            plt.xlabel('$k_{B}T$ (in units of $J$)')
            plt.ylabel('$c_{V}(k_{B}T)$')
            plt.legend()
            if args.save:
                plt.savefig(f'P2-1{args.part}.png', dpi=1200)
            else:
                plt.title('Specific heat per particle at constant volume over 1,000 trial spin \n '
                          f'flips, after equilibrium has been reached, for {simulations:,.0f} simulations each')
                plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default=False, action="store_true", help="Save plots generated instead of showing.")
    parser.add_argument('--part', type=str, default='a', help="Choose the code for the given part to be executed.")
    args = parser.parse_args()
    main()
