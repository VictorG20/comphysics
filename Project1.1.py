import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


rng = np.random.default_rng(42)


def mc_integrator():
    int_volume = args.b - args.a
    points = int_volume * rng.random(args.N) + args.a
    function = points ** 4
    avg_function = np.sum(function) / args.N
    avg_sqr_function = np.sum(function * function) / args.N
    mc_integral = int_volume * avg_function
    std_error = int_volume * np.sqrt((avg_sqr_function - avg_function ** 2) / (args.N - 1))
    return mc_integral, std_error


def main():
    integral = []
    error = []
    for i in range(args.M):
        mc_integral, std_error = mc_integrator()
        integral.append(mc_integral)
        error.append(std_error)
        if not args.show:
            continue
        print("The Monte-Carlo integrator gives:", mc_integral, "with a standard error of", std_error)

    plt.hist(integral, bins=80, density=True, ec='black', lw=0.5, color='dodgerblue', label='Data generated')

    # Gaussian fit
    avg, std = norm.fit(integral)  # The function 'norm.fit' takes simply the mean and standard deviation of the data.
    x_min, x_max = plt.xlim()
    x = np.linspace(x_min, x_max, 100)
    gauss_fit = norm.pdf(x, avg, std)

    plt.plot(x, gauss_fit, color='r', lw=3, label=f'Gaussian fit: \n $\\mu = ${avg:.2f}, $\\sigma = ${std:.2f}')
    plt.xlabel('Value for $I_{N}$')
    plt.ylabel('Frequency')
    plt.title('$I_{N}$ results for ' + str(args.N) + ' different sets of random numbers')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=0, help="Lower limit of the integral")
    parser.add_argument('--b', type=float, default=1, help="Upper limit of the integral")
    parser.add_argument('--N', type=int, default=1000, help="Number of points for a single integration")
    parser.add_argument('--M', type=int, default=1000, help='Number of sets of random numbers')
    parser.add_argument('--show', default=False, action="store_true", help="Show values for integral and error")
    parser.add_argument('--sampling', default=False, action='store_true', help='Use importance sampling')
    args = parser.parse_args()
    main()
