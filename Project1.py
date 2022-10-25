import argparse
import numpy as np


def main():
    rng = np.random.default_rng(42)
    int_volume = args.b - args.a
    points = int_volume * rng.random(args.N) + args.a
    function = points**4
    avg_function = np.sum(function)/args.N
    avg_sqr_function = np.sum(function*function)/args.N
    mc_integral = int_volume*avg_function
    std_error = int_volume*np.sqrt((avg_sqr_function - avg_function**2)/(args.N - 1))
    print("The Monte-Carlo integrator gives:", mc_integral, "with a standard error of", std_error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=float, default=0, help="Lower limit of the integral")
    parser.add_argument('--b', type=float, default=1, help="Upper limit of the integral")
    parser.add_argument('--N', type=int, default=1000, help="Number of points")
    args = parser.parse_args()
    main()
