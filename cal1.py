import numpy as np

# PART A: Define the function and its derivatives
def f1(x):
    return np.exp(2 * np.sin(x)) - 2 * x - 1

def df1(x):
    return 2 * np.exp(2 * np.sin(x)) * np.cos(x) - 2

def ddf1(x):
    sinx = np.sin(x)
    cosx = np.cos(x)
    exp_term = np.exp(2 * sinx)
    return 2 * exp_term * (-np.sin(x) + 2 * cosx**2)

# Verify multiplicity-2 root at x = 0
x_check = 0
print("Part A: Verifying root of multiplicity 2 at x = 0")
print(f"f(0) = {f1(x_check)}")
print(f"f'(0) = {df1(x_check)}")
print(f"f''(0) = {ddf1(x_check)}\n")

# General Newton’s method
def newton_method(f, df, x0, steps=9):
    x = x0
    for _ in range(steps):
        x = x - f(x)/df(x)
    return x

# Modified Newton’s method (for multiplicity 2)
def modified_newton_method(f, df, x0, steps=9):
    x = x0
    for _ in range(steps):
        x = x - 2 * f(x)/df(x)
    return x

# PART B: Apply methods to f1 starting from x0 = 0.1
x0 = 0.1
x9_newton_f1 = newton_method(f1, df1, x0)
x9_modified_f1 = modified_newton_method(f1, df1, x0)

print("Part B: Newton’s Method vs Modified Newton’s Method on f(x) = e^(2sinx) - 2x - 1")
print(f"x9 (Newton's)      = {x9_newton_f1}")
print(f"x9 (Modified)      = {x9_modified_f1}")
print()

# PART C: f(x) = (8x^2)/(3x^2 + 1)
def f2(x):
    return (8 * x**2) / (3 * x**2 + 1)

def df2(x):
    numerator = 16 * x * (1 + x**2)
    denominator = (3 * x**2 + 1)**2
    return numerator / denominator

x0_f2 = 0.15
x9_newton_f2 = newton_method(f2, df2, x0_f2)
x9_modified_f2 = modified_newton_method(f2, df2, x0_f2)

print("Part C: Newton’s Method vs Modified Newton’s Method on f(x) = (8x^2)/(3x^2 + 1)")
print(f"x9 (Newton's)      = {x9_newton_f2}")
print(f"x9 (Modified)      = {x9_modified_f2}")
