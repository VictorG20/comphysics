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
    ax.set_xticks(np.arange(1, spins+1, 1), labels='')
    ax.set_yticks(np.arange(0, 2*np.size(save_trials), 2), labels=save_trials)
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
    ax2.set_yticks(np.arange(0, 2*np.size(save_trials), 2), labels=down_spins)
    ax2.set_ylabel('Amount of spins down')
    ax2.set_aspect(0.9)
    plt.tight_layout()

    if args.save:
        plt.savefig(f'P2-1{args.part}.png', dpi=1200)
    else:
        plt.title('System at the given trials')
        plt.show()
    return


def trial_spin_flips(state, thermal_energy, flips, h=0.):
    spins = np.size(state)
    energy = -1. * np.sum([state[i]*state[(i+1) % spins] for i in range(spins)]) - h*np.sum(state)
    energies = [energy]  # Keep track of the energy at each trial.
    magnetisation = np.sum(state)
    magnetisation_values = [magnetisation]

    energy_differences = sorted(set(2.*spin_mid*(spin_left + spin_right + h) for spin_mid in [-1, 1]
                                    for spin_left in [-1, 1] for spin_right in [-1, 1]))
    pos_energy_differences = [delta_E for delta_E in energy_differences if delta_E > 0]
    acceptance_prob = [np.exp(-delta_E / thermal_energy) for delta_E in pos_energy_differences]

    trials = rng.integers(low=0, high=spins, size=flips)  # Chosen spin sites to try to flip. 'high' is exclusive.

    for trial in trials:
        energy_difference = 2.*state[trial]*(state[(trial + 1) % spins] + state[trial - 1] + h)  # Modulo due to P.B.C.
        if energy_difference > 0 and rng.random(1) > acceptance_prob[pos_energy_differences.index(energy_difference)]:
            energies.append(energy)
            magnetisation_values.append(magnetisation)
            continue
        state[trial] = -1 * state[trial]  # Flip spin
        magnetisation += 2.*state[trial]  # Update magnetisation of the system (with new value of the spin flipped)
        energy += energy_difference  # Update energy of the system
        energies.append(energy)
        magnetisation_values.append(magnetisation)
    return energies, magnetisation_values


def part_a(spins):
    state = [-1 for _ in range(spins)]  # All spins initially pointing in the same direction. Here it is downwards.
    thermal_energy = 1.
    save_trials = [0, 5, 50, 100, 200, 300, 350, 400, 450, 500]  # Chosen times to visualize the system.
    saved_states = [state.copy()]  # Save states at the given 'save_trials' numbers to be plotted later.
    flips_done = 0
    for number in save_trials:
        if number == 0:
            continue
        trial_spin_flips(state, thermal_energy, flips=number-flips_done)
        saved_states.append(state.copy())
        flips_done = number
    visualize_states(save_trials, saved_states)
    return


def part_b(spins):
    thermal_energies = [0.1, 1, 10]
    flips = 1000
    colors = ['darkviolet', 'orange', 'red']
    for i in range(np.size(thermal_energies)):
        state = [-1 for _ in range(spins)]  # All spins initially pointing in the same direction (here downwards).
        energies, _ = trial_spin_flips(state, thermal_energies[i], flips)
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
            energies, _ = trial_spin_flips(state, thermal_energies[i], flips)
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


def analytical_magnetisation(kbt, h):
    m = np.exp(1./kbt) * np.sinh(h/kbt) / np.sqrt(np.exp(2./kbt) * (np.sinh(h/kbt)**2.) + np.exp(-2./kbt))
    return m


def main():
    if args.part == 'a':
        part_a(spins=20)
    elif args.part == 'b':
        part_b(spins=20)
    elif args.part == 'c' or args.part == 'd':
        spins = 20  # Number of spins in the system
        thermal_energies = [i+1 for i in range(10)]  # Choose kT = 1, 2, ..., 10
        simulations = 100  # Number of independent simulations for each thermal energy
        avg_energy_particle = []
        specific_heat_particle = []
        for thermal_energy in thermal_energies:
            time_avg_energies = []
            specific_heats = []
            for _ in range(simulations):
                state = [-1 for _ in range(spins)]
                trial_spin_flips(state, thermal_energy, flips=1000)  # Run the initial state for 1000 flips first.
                energies, _ = trial_spin_flips(state, thermal_energy, flips=1000)  # Now get the energies after eq.
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
    elif args.part == 'e':
        spins = 20
        thermal_energies = [i + 1. for i in range(10)]  # Choose kT = 1, 2, ..., 10
        simulations = 100  # Number of independent simulations for each thermal energy
        h_values = [0., 0.1, 1., 10.]
        temperatures = np.linspace(np.min(thermal_energies), np.max(thermal_energies), num=100)
        colors = ['red', 'salmon', 'dodgerblue', 'deepskyblue', 'forestgreen', 'limegreen', 'darkviolet', 'violet']
        for h_field in h_values:
            magnetisation_particle = []
            for thermal_energy in thermal_energies:
                simulations_avg_magnetisation = []
                for simulation in range(simulations):
                    state = [-1 for _ in range(spins)]
                    trial_spin_flips(state, thermal_energy, flips=1000, h=h_field)  # Reach equilibrium
                    _, magnetisation_values = trial_spin_flips(state, thermal_energy, flips=1000, h=h_field)
                    simulations_avg_magnetisation.append(np.mean(magnetisation_values))
                magnetisation_particle.append(np.mean(simulations_avg_magnetisation)/spins)
            plt.plot(thermal_energies, magnetisation_particle, 'o', color=colors[2*h_values.index(h_field)],
                     label=f'$H =$ {h_field}')
            plt.plot(temperatures, analytical_magnetisation(temperatures, h_field),
                     color=colors[2*h_values.index(h_field) + 1])
        ax = plt.gca()
        ax.set_xticks(np.arange(np.min(thermal_energies), np.max(thermal_energies)+1, 1))
        plt.xlabel('$k_{B}T$ (in units of $J$)')
        plt.ylabel('$m = \\langle M \\rangle / N$')
        plt.legend()
        if args.save:
            plt.savefig(f'P2-1{args.part}.png', dpi=1200)
        else:
            plt.title('Mean magnetization per particle over 1,000 flip spin trials after \n '
                      f'equilibrium has been reached and over {simulations:,.0f} simulations')
            plt.show()
    elif args.part == 'f':
        spins = 20
        state = rng.choice([-1, 1], size=spins)
        print(state)
    else:
        print('Not a valid part of the project. Only options available are from a to f.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default=False, action="store_true", help="Save plots generated instead of showing.")
    parser.add_argument('--part', type=str, default='a', help="Choose the code for the given part to be executed.")
    args = parser.parse_args()
    main()
