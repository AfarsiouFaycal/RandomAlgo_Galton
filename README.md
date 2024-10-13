# Prerequisites

To compile and run this program, you need:

    Python 3.x
    Required Python packages:
        numpy
        matplotlib
        seaborn
        scipy


# How to Run the Program
## Simulating the Galton Board

The code simulates a Galton board where balls drop through multiple levels, taking random paths to the bottom. To simulate the Galton board with 10 levels and 1000 balls, run the following command in the Python environment:
```
n = 10      # Number of levels
num_balls = 1000  # Number of balls to drop

# Run the simulation
triangle = simulate_galton_board(n, num_balls)

# Plot the results
plot_histogram(triangle, n, num_balls)
plot_normal(n, triangle, num_balls)
```

## Effect of Increasing n and Number of Balls

The code also allows you to explore the effect of increasing both the number of levels (n) and the number of balls dropped (N), showing how the experimental results compare to theoretical distributions:

    effect_of_n_and_N(Two_ns=[10, 100], Two_Ns=[1000, 100000])

This will plot graphs comparing the experimental results with both the binomial and normal distributions for different values of n and N.

## Visualizing the Convergence to the Normal Distribution

You can visualize the mean squared error between the experimental data and normal distribution N(μ,σ²) with μ and σ the mean and standard deviation of the experimental data, for different values of n and different numbers of balls:
```
mean_diff_plots_exp_norm()
mean_diff_plots_exp_bin()
```
This will generate a graph showing how the error decreases as more balls are dropped and the size of the board increases.


## Evaluating the Approximation of Binomial to Normal Distribution

This script allows you to evaluate how the binomial distribution Bin(n, 1/2) converges to the normal distribution N(n/2, n/4) as n grows. Run the following to generate a comparison graph:

    mean_diff_plots_norm_bin()

This will generate a graph illustrating the mean squared error between the binomial distribution and the normal approximation as the number of levels (n) grows.
