import scipy
import numpy as np
from scipy.optimize import LinearConstraint, minimize, NonlinearConstraint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from itertools import combinations
from math import comb
import plotly.graph_objects as go
from functools import partial
from multiprocessing import Pool
import time
import json

# Global variables
n = 4
tol = 1e-8
points_len = 100
points_width = 100


symbols = sp.symbols('a_:'+str(n**2))
matrix = sp.Matrix(n,n, symbols)
string_matrix = [['' for i in range(n)] for j in range(n)]

m=0
A = np.zeros((n, comb(n, 2)))
for j in range(n):   
    for k in range(j,n):
        if k == j:
            matrix[j,k] = 'a_16'
            continue
        m+=1
        string_matrix[j][k] = '-a_'+str(m)
        string_matrix[k][j] = '-a_'+str(m)
        matrix[j,k] = 'a_'+str(m)
        matrix[k,j] = 'a_'+str(m)
        A[j][m-1] = 1
        A[k][m-1] = 1

    row_sum = '1'
    for k in range(n):
        row_sum += string_matrix[j][k]
    matrix[j,j] = row_sum

matrix_constraints = LinearConstraint(A, np.zeros(n), np.ones(n))

def sum_matrix_minors(matrix, k):
    return sum(matrix[i, i].det() for i in combinations(range(n), k))

def run_function_with_const(loc, constraints = matrix_constraints):
    bounds = [(0.0, 1.0)] * comb(n, 2)

    count = 0
    best_result = None
    num_starts = 140
    while count < 30:
        results = [minimize(funcs_of_principal_minors[loc], np.random.rand(comb(n, 2)), bounds=bounds, constraints=constraints, method='trust-constr', tol=tol, options={'maxiter': 300, 'initial_constr_penalty': 10000}) for _ in range(num_starts)]

        best_result = results[0]
        is_false = False
        for result in results:
            if result.success:
                is_false = True
                best_result = result
                break

        if is_false:
            for result in results:
                if result.fun <= best_result.fun and result.success:
                    best_result = result
            return best_result
        else:
            count += 1
            num_starts = 300

    print(count)
    return best_result

def optimize_func(loc, eqs = []):
    if len(eqs) == 0:
        return run_function_with_const(loc)
    equals = []
    for i in range(0, len(eqs)):
        equals.append(NonlinearConstraint(
        lambda x: funcs_of_principal_minors[eqs[i][0]](x) - eqs[i][1],
        [0.0],
        [0.0]
    ))
    equals.append(matrix_constraints)
    return run_function_with_const(loc, equals)

funcs_of_principal_minors = tuple(
    sp.lambdify([symbols[1:comb(n, 2)+1]], sum_matrix_minors(matrix, k), 'numpy')
    for k in range(1, n+1)
) + tuple(
    sp.lambdify([symbols[1:comb(n, 2)+1]], -1*sum_matrix_minors(matrix, k), 'numpy')
    for k in range(1, n+1)
)

def compute_min_y(x):
    return optimize_func(1, [[0, x]])

def compute_max_y(x):
    results = optimize_func(1 + n, [[0 + n, -x]])
    results.fun *= -1
    return results

def compute_min_z(point):
    return optimize_func(2, [[0,point[0]], [1,point[1]]])

def compute_max_z(point):
    results = optimize_func(2+n, [[0+n,-point[0]], [1+n,-point[1]]])
    results.fun *= -1
    return results

def convert_optimize_result_to_dict(result):
        """Converts a scipy OptimizeResult object to a dictionary."""
        if result is None:
            return None
        
        result_dict = {
            'matrix': result.x.tolist() if isinstance(result.x, np.ndarray) else result.x,
            'success': result.success,
            'output': result.fun,
            'message': result.message
        }
        return result_dict

def save_optimization_results(results_x, results_y, results_z, filename="optimization_results.json"):
    """Saves a list of optimization results to a JSON file."""
    results_z = [convert_optimize_result_to_dict(result) for result in results_z]

    results = [res for res in zip(results_x, results_y, results_z)]

    with open(filename, 'w') as f:
            json.dump(results, f, indent=4)

def save_spectra_results(results_x, results_y, results_z, filename="spectra_results.json"):
    string_results = []
    for point in zip(results_x, results_y, results_z):
        dict_val = {
            'x': str(point[0]),
            'y': str(point[1]),
            'z': str(point[2])
        }
        string_results.append(dict_val)
    with open(filename, 'w') as f:
            json.dump(string_results, f, indent=4)

if __name__ == '__main__':
    start_time = time.perf_counter()
    x_values = np.linspace(0, n, points_len)
    with Pool() as pool:
        min_y_values = pool.map(compute_min_y, x_values)
        max_y_values = pool.map(compute_max_y, x_values)

    X_mesh = []
    Y_mesh = []

    for i in range(len(x_values)):
        y_vals = np.linspace(min_y_values[i].fun, max_y_values[i].fun, points_width)
        X_mesh.append(np.full_like(y_vals, x_values[i]))
        Y_mesh.append(y_vals)

    X = np.concatenate(X_mesh)
    Y = np.concatenate(Y_mesh)

    print("XY Done")

    with Pool() as pool:
        Z_min = pool.map(compute_min_z, zip(X,Y))
        Z_max = pool.map(compute_max_z, zip(X,Y))

    print(f"Time taken to get coef data: {time.perf_counter() - start_time:.6f} seconds")

    save_optimization_results(X, Y, Z_min, "ds-sniep_min_values_4.json")
    save_optimization_results(X, Y, Z_max, "ds-sniep_max_values_4.json")

    print("data saved.")
