import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

rng = np.random.default_rng(42)


'''
For part a) I need to:
    Print the values of I_{N} and sigma_{N}
For part b) I need to:
    Parameter M = 1000
    Fit a Gaussian distribution
    Plot a histogram and the Gaussian fitting curve
For part c) I need to:
    Have a sampling distribution g(x) different from '1'
    For each g(x):
        Repeat the steps from part b) including the plot
        Make a log-log plot of sigma_{N} vs. N

In general, I always have:
    The function to be integrated: f(x) = x^4
    Parameter N = 1000
   
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

    plt.plot(x, gauss_fit, color='r', lw=3, label=f'Gaussian fit: \n $\\mu = ${avg:.2f}, $\\sigma = ${std:.2f}')
    plt.xlabel('Value for $I_{N}$')
    plt.ylabel('Frequency')
    plt.title('$I_{N}$ results for ' + str(args.N) + ' different sets of random numbers')
    plt.legend()
    return plt.show()


def main():
    if not args.part_c:
        def sampling_distribution(_):
            return 1
        if args.part_a:
            mc_integral, std_error = mc_integrator(sampling_distribution)
            print("The Monte-Carlo integrator gives:", mc_integral, "with a standard error of", std_error)
            sys.exit()

        integral = [mc_integrator(sampling_distribution)[0] for _ in range(args.M)]
        histogram_and_gaussian(integral)

    def sampling_distribution(interval):
        return 5*interval**4

    integral = [mc_integrator(sampling_distribution)[0] for _ in range(args.M)]
    histogram_and_gaussian(integral)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=0, help="Lower limit of the integral")
    parser.add_argument('--b', type=float, default=1, help="Upper limit of the integral")
    parser.add_argument('--N', type=int, default=1000, help="Number of points for a single integration")
    parser.add_argument('--M', type=int, default=1000, help='Number of sets of random numbers')
    parser.add_argument('--part_a', default=False, action="store_true", help="Obtain results for part a)")
    parser.add_argument('--part_b', default=True, action="store_true", help="Obtain results for part b)")
    parser.add_argument('--part_c', default=False, action="store_true", help="Obtain results for part c)")
    args = parser.parse_args()
    main()
