import numpy as np
from scipy.optimize import minimize_scalar

# Given values
N = 100
q = 0.9

# Define the function to minimize
def average_tests(x):
    return N * (1 - q**x + 1/x)

# Minimize the function in the range [1, 150]
result = minimize_scalar(average_tests, bounds=(1, 150), method='bounded')

# Output results
print("Optimal group size x:", result.x)
print("Minimum average number of tests:", result.fun)
