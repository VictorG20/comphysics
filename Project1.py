import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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


def gaussian(x, A, B):
    # norm_factor = 1/np.sqrt(2*np.pi * std**2)
    y = A*np.exp(-1*B*(x-0.2)**2)
    return y


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
    counts, edges = np.histogram(integral, bins=100)
    parameters, _ = curve_fit(gaussian, edges[:-1], counts)  # Check 'norm.fit' function from scipy stats
    fit_A = parameters[0]
    fit_B = parameters[1]
    gaussian_fit = gaussian(edges[:-1], fit_A, fit_B)
    plt.bar(edges[:-1], counts, width=np.diff(edges))
    plt.plot(edges[:-1], gaussian_fit)
    # plt.xlabel('Value for $I_{N}$')
    # plt.ylabel('Frequency')
    # plt.title('$I_{N}$ results for ' + str(args.N) + ' different sets of random numbers')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=0, help="Lower limit of the integral")
    parser.add_argument('--b', type=float, default=1, help="Upper limit of the integral")
    parser.add_argument('--N', type=int, default=1000, help="Number of points")
    parser.add_argument('--M', type=int, default=1, help='Number of sets of random numbers')
    parser.add_argument('--show', default=False, action="store_true", help="Show values for integral and error")
    args = parser.parse_args()
    main()
