from scipy.optimize import differential_evolution
import numpy as np

def simple_func(x):
    return np.sum(x**2)

bounds = [(0, 1)] * 2  # Example with 2 variables
result = differential_evolution(simple_func, bounds)
print(result)