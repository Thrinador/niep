import tomli
import time
import json
from multiprocessing import Pool
from coef_optimizer import *

with open("config.toml", "rb") as f:
    data = tomli.load(f)

points_dim = data['global_data']['points_dim']

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

if __name__ == '__main__':
    start_time = time.perf_counter()
    x_values = np.linspace(0, n, points_dim[0])
    with Pool() as pool:
        min_y_values = pool.map(compute_min_y, x_values)
        max_y_values = pool.map(compute_max_y, x_values)

    X, Y = build_XY_mesh(x_values, min_y_values, max_y_values)

    print(f"XY done in: {time.perf_counter() - start_time:.6f} seconds")
    start_time = time.perf_counter()

    with Pool() as pool:
        Z_min = pool.map(compute_min_z, zip(X,Y))
        print(f"Z_min done in: {time.perf_counter() - start_time:.6f} seconds")
        Z_max = pool.map(compute_max_z, zip(X,Y))

    print(f"Time taken to get coef data: {time.perf_counter() - start_time:.6f} seconds")
         
    save_optimization_results(X, Y, Z_min, build_file_name(False))
    save_optimization_results(X, Y, Z_max, build_file_name(True))

    print("data saved.")
