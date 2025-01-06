import numpy as np
from scipy.optimize import LinearConstraint, minimize, NonlinearConstraint
import sympy as sp
from itertools import combinations
from math import comb
import tomli
import time
import json
from pathos.pools import ProcessPool as Pool

# Build config variables
with open("coef_config.toml", "rb") as f:
    data = tomli.load(f)
points_dim = data['global_data']['points_dim']
n = data['global_data']['n']
tol = data['global_data']['tol']
initial_runs = data['global_data']['initial_runs']
subsequent_runs = data['global_data']['subsequent_runs']
type = data['global_data']['type']
num_variables = n**2 - n if type == 0 else comb(n, 2)
funcs_to_optimize = data['global_data']['funcs_to_optimize']

# Setup symbolic matrix to be working with.
symbols = sp.symbols('a_:'+str(n**2))
matrix = sp.Matrix(n,n, symbols)
string_matrix = [['' for i in range(n)] for j in range(n)]

A = np.zeros((n, num_variables))
m=0
if type == 0:
    for j in range(n):   
        for k in range(n):
            if k == j:
                matrix[j,k] = 'a_16'
                continue
            string_matrix[j][k] = '-a_'+str(m)
            matrix[j,k] = 'a_'+str(m)
            A[j][m-1] = 1
            m+=1

        row_sum = '1'
        for k in range(n):
            row_sum += string_matrix[j][k]
        matrix[j,j] = row_sum
else:
    for j in range(n):   
        for k in range(j,n):
            if k == j:
                matrix[j,k] = 'a_16'
                continue
            string_matrix[j][k] = '-a_'+str(m)
            string_matrix[k][j] = '-a_'+str(m)
            matrix[j,k] = 'a_'+str(m)
            matrix[k,j] = 'a_'+str(m)
            A[j][m] = 1
            A[k][m] = 1
            m+=1

        row_sum = '1'
        for k in range(n):
            row_sum += string_matrix[j][k]
        matrix[j,j] = row_sum

matrix_constraints = LinearConstraint(A, np.zeros(n), np.ones(n))

def sum_matrix_minors(matrix, k):
    return sum(matrix[i, i].det() for i in combinations(range(n), k))

def run_function_with_const(loc, constraints = matrix_constraints):
    bounds = [(0.0, 1.0)] * num_variables

    count = 0
    best_result = None
    num_starts = initial_runs
    while count < 30:
        results = [minimize(funcs_of_principal_minors[loc], np.random.rand(num_variables), bounds=bounds, constraints=constraints, method='trust-constr', tol=tol, options={'maxiter': 300, 'initial_constr_penalty': 10000}) for _ in range(num_starts)]

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
            num_starts = subsequent_runs

    print(count)
    return best_result

def optimize_func(loc, eqs = []):
    if len(eqs) == 0:
        return run_function_with_const(loc)
    equals = [NonlinearConstraint(lambda x, j=i: funcs_of_principal_minors[eqs[j][0]](x) - eqs[j][1], 0,0) for i in range(len(eqs))]
    equals.append(matrix_constraints)
    return run_function_with_const(loc, equals)

funcs_of_principal_minors = tuple(
    sp.lambdify([symbols[0:num_variables]], sum_matrix_minors(matrix, k+1), 'numpy')
    for k in range(n)
) + tuple(
    sp.lambdify([symbols[0:num_variables]], -1*sum_matrix_minors(matrix, k+1), 'numpy')
    for k in range(n)
)

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

def save_optimization_results(results_x, results_y, results_z=[], filename="optimization_results.json"):
    """Saves a list of optimization results to a JSON file."""
    if results_z == []:
        results_y = [convert_optimize_result_to_dict(result) for result in results_y]
        results = [res for res in zip(results_x, results_y)]
        with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
    else:
        results_z = [convert_optimize_result_to_dict(result) for result in results_z]
        results = [res for res in zip(results_x, results_y, results_z)]
        with open(filename, 'w') as f:
                json.dump(results, f, indent=4)

def build_file_name(is_max):
    type = "niep/" if data['global_data']['type'] == 0 else "ds-sniep/"
    base = data['global_data']['save_location']
    if base == "":
        base = "niep" if data['global_data']['type'] == 0 else "ds-sniep"
    maxmin = "_max_values_" if is_max else "_min_values_"
    runs = ""
    for dim in points_dim:
         runs += "_" + str(dim)
    return "data/" + type + base + maxmin + str(n) + runs + ".json"

def build_XY_mesh(x_values, min_y_values, max_y_values):
    X_mesh = []
    Y_mesh = []

    for i in range(len(x_values)):
        y_vals = np.linspace(min_y_values[i].fun, max_y_values[i].fun, points_dim[1])
        X_mesh.append(np.full_like(y_vals, x_values[i]))
        Y_mesh.append(y_vals)

    return np.concatenate(X_mesh), np.concatenate(Y_mesh)

def optimize_first_func(constraint_loc=0, func_loc=1):
    x_values = np.linspace(0, n, points_dim[0])
    with Pool() as pool:
        min_y_values = pool.map(lambda x: optimize_func(func_loc, [[constraint_loc, x]]), x_values)
        max_y_values = pool.map(lambda x: optimize_func(func_loc + n, [[constraint_loc + n, -x]]), x_values)
        for max_y_val in max_y_values:
            max_y_val.fun *= -1
    return x_values, min_y_values, max_y_values

def optimize_second_func(X, Y, constraint_loc_1=0, constraint_loc_2=1, func_loc=2):
    with Pool() as pool:
        Z_min = pool.map(lambda point: optimize_func(func_loc, [[constraint_loc_1,point[0]], [constraint_loc_2,point[1]]]), zip(X,Y))
        Z_max = pool.map(lambda point: optimize_func(func_loc+n, [[constraint_loc_1+n,-point[0]], [constraint_loc_2+n,-point[1]]]), zip(X,Y))
        for max_z_val in Z_max:
            max_z_val.fun *= -1
    return Z_min, Z_max

if __name__ == '__main__':
    print("Starting first func optimization.")
    start_time = time.perf_counter()
    x_values, min_y_values, max_y_values = optimize_first_func(func_loc=funcs_to_optimize[0])
    print(f"First func optimized in {time.perf_counter() - start_time:.6f} seconds")

    if len(funcs_to_optimize) > 1:
        print("Starting second func optimization.")
        start_time = time.perf_counter()
        X, Y = build_XY_mesh(x_values, min_y_values, max_y_values)
        Z_min, Z_max = optimize_second_func(X,Y)
        print(f"Second func optimized in {time.perf_counter() - start_time:.6f} seconds")
        save_optimization_results(X, Y, Z_min, build_file_name(False))
        save_optimization_results(X, Y, Z_max, build_file_name(True))
    else:
        print("No second function, saving data")
        save_optimization_results(x_values, min_y_values, build_file_name(False))
        save_optimization_results(x_values, max_y_values, build_file_name(True))

    print("data saved.")
