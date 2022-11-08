import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

rng = np.random.default_rng(42)


'''
For part c) I need to:
    Have a sampling distribution g(x) different from '1'
    For each g(x):
        Repeat the steps from part b) including the plot
        Make a log-log plot of sigma_{N} vs. N
      
    Save the values of the mean and standard deviation for each sampling distribution and fit a Gaussian
'''


def mc_integrator(sampling_distribution):
    int_volume = args.b - args.a
    points = int_volume * rng.random(args.N) + args.a
    function = points ** 4
    g = sampling_distribution(points)
    avg_function = np.sum(function * g) / args.N
    avg_sqr_function = np.sum(function * function * g) / args.N
    mc_integral = int_volume * avg_function
    std_error = int_volume * np.sqrt((avg_sqr_function - avg_function ** 2) / (args.N - 1))
    return mc_integral, std_error


def histogram_and_gaussian(data):
    plt.hist(data, bins=80, density=True, ec='black', lw=0.5, color='dodgerblue', label='Data generated')

    # Gaussian fit
    avg, std = norm.fit(data)  # 'norm.fit' takes simply the mean and standard deviation of the data.
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 100)
    gauss_fit = norm.pdf(x, avg, std)

    plt.plot(x, gauss_fit, color='r', lw=3, label=f'Gaussian fit: \n $\\mu = ${avg:.3f}, $\\sigma = ${std:.3f}')
    plt.xlabel('Value for $I_{N}$')
    plt.ylabel('Frequency')
    plt.title('$I_{N}$ results for ' + str(args.M) + ' different sets of random numbers')
    plt.legend()
    if not args.save:
        plt.show()
    elif args.b:
        plt.savefig('P1-1b.png', dpi=1200)
    else:
        plt.savefig('P1-1c.png', dpi=1200)
    return


def importance_sampling_1():
    averages = []
    std_dev = []
    sampling_power = [1, 2, 3, 4]
    for power in sampling_power:
        integral_value = []
        std_errors = []
        for _ in range(args.M):
            numbers = rng.random(args.N)
            x = np.power(numbers, 1./(power+1))
            function = x ** 4
            sampling_distribution = (power + 1) * x**power
            avg_function = np.sum(np.divide(function, sampling_distribution))/args.N
            avg_sqr_function = np.sum(np.divide(function * function, sampling_distribution))/args.N
            std_error = np.sqrt((avg_sqr_function - avg_function ** 2) / (args.N - 1))
            integral_value.append(avg_function)
            std_errors.append(std_error)
        averages.append(np.mean(integral_value))
        std_dev.append(np.mean(std_errors))
    return averages, std_dev


def importance_sampling_2():
    sampling_power = [1, 2, 3, 4]
    for power in sampling_power:
        std_dev = []
        n_values = []
        for n in range(10, 1000, 10):
            errors = []
            for _ in range(10):
                numbers = rng.random(n)
                x = np.power(numbers, 1. / (power + 1))
                function = x ** 4
                sampling_distribution = (power+1) * x ** power
                avg_function = np.sum(np.divide(function, sampling_distribution)) / n
                avg_sqr_function = np.sum(np.divide(function * function, sampling_distribution)) / n
                std_error = np.sqrt((avg_sqr_function - avg_function ** 2) / (n - 1))
                errors.append(std_error)
            std_dev.append(np.mean(errors))
            n_values.append(n)
        plt.loglog(n_values, std_dev, label=f'$g(x) = {power+1} x^{power}$')
    plt.xlabel('$\\log{N}$')
    plt.ylabel('$\\log{\\sigma_{N}}$')
    plt.legend()
    plt.show()
    return


def main():
    if not args.part_c:
        def sampling_distribution(_):  # For cases (a) and (b) the sampling distribution is 1.
            return 1
        if args.part_a:
            mc_integral, std_error = mc_integrator(sampling_distribution)
            print("The Monte-Carlo integrator gives:", mc_integral, "with a standard error of", std_error)
            sys.exit()

        integral = [mc_integrator(sampling_distribution)[0] for _ in range(args.M)]
        histogram_and_gaussian(integral)
        sys.exit()

    print(importance_sampling_1())
    importance_sampling_2()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=0, help="Lower limit of the integral")
    parser.add_argument('--b', type=float, default=1, help="Upper limit of the integral")
    parser.add_argument('--N', type=int, default=1000, help="Number of points for a single integration")
    parser.add_argument('--M', type=int, default=1000, help='Number of sets of random numbers')
    parser.add_argument('--part_a', default=False, action="store_true", help="Obtain results for part a)")
    parser.add_argument('--part_b', default=True, action="store_true", help="Obtain results for part b)")
    parser.add_argument('--part_c', default=False, action="store_true", help="Obtain results for part c)")
    parser.add_argument('--save', default=False, action="store_true", help="Save plots generated.")
    args = parser.parse_args()
    main()
