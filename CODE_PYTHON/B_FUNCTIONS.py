import numpy as np

def sphere_function(x):
    return np.sum(x**2)

def quartic_function(x):
    return np.sum(np.arange(1, len(x) + 1) * (x ** 4))

def rastrigin_function(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def ackley_function(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x).sum())) + 20 + np.e

def rosenbrock_function(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def griewank_function(x):
    return 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

def schwefel_function(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def michalewicz_function(x, m = 10):
    return -np.sum(np.sin(x) * np.sin(np.arange(1, len(x) + 1) * x**2 / np.pi)**(2 * m))

def rastrigin_modified_function(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Ackley Function
def ackley_function(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x).sum())) + 20 + np.e

# Alpine Function
def alpine_function(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

# Brown Function
def brown_function(x):
    return np.sum((x[:-1]**2) ** (x[1:]**2 + 1) + (x[1:]**2) ** (x[:-1]**2 + 1))

# Dixon and Price Function
def dixon_price_function(x):
    return (x[0] - 1)**2 + np.sum(np.arange(2, len(x) + 1) * (2 * x[1:]**2 - x[:-1])**2)

# Griewank Function
def griewank_function(x):
    return 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

# Pathological Function
def pathological_function(x):
    term1 = 0.5
    term2 = np.sin(np.sqrt(100 * x[:-1]**2 + x[1:]**2))**2 - 0.5
    term3 = (1 + 0.001 * (x[:-1]**2 - 2 * x[:-1] * x[1:] + x[1:]**2))**2
    return np.sum(term1 + term2 / term3)

# Quartic Function
def quartic_function(x):
    return np.sum(np.arange(1, len(x) + 1) * (x ** 4))

# Rosenbrock's Function
def rosenbrock_function(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Rosenbrock and Yang's Function
def rosenbrock_yang_function(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# Rotated Hyper-Ellipsoid Function
def rotated_hyper_ellipsoid_function(x):
    return np.sum([np.sum(x[:i+1]**2) for i in range(len(x))])

# Schumer Steiglitz Function
def schumer_steiglitz_function(x):
    return np.sum(x**4)

# Schwefel 2 Function
def schwefel2_function(x):
    return np.sum(np.abs(x))

# Schwefel 3 Function
def schwefel3_function(x):
    return np.sum([np.sum(x[:i+1])**2 for i in range(len(x))])

# Schwefel 4 Function
def schwefel4_function(x):
    return np.max(np.abs(x))

# Step Function
def step_function(x):
    return np.sum(np.floor(x + 0.5)**2)

# Trigonometric Function
def trigonometric_function(x):
    return np.sum(1 - np.cos(x) + 0.1 * (1 - np.cos(2 * x)))

# Zakharov Function
def zakharov_function(x):
    term1 = np.sum(x**2)
    term2 = (0.5 * np.arange(1, len(x) + 1) * x).sum()**2
    term3 = (0.5 * np.arange(1, len(x) + 1) * x).sum()**4
    return term1 + term2 + term3

