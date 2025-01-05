from library_tools_ds_sniep import *

def compute_min_y(x):
    return optimize_func(2, [[0, x]])

def compute_max_y(x):
    results = optimize_func(2 + n, [[0 + n, -x]])
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

def save_optimization_results(results, filename="optimization_results.json"):
    """Saves a list of optimization results to a JSON file."""
    results_dict_list = [convert_optimize_result_to_dict(result) for result in results]

    with open(filename, 'w') as f:
            json.dump(results_dict_list, f, indent=4)

if __name__ == '__main__':
    start_time = time.perf_counter()
    x_values = np.linspace(0, n, 1000)
    
    with Pool() as pool:
        min_y_values = pool.map(compute_min_y, x_values)
        max_y_values = pool.map(compute_max_y, x_values)


    print(f"Time taken to get coef data: {time.perf_counter() - start_time:.6f} seconds")

    save_optimization_results(min_y_values, "min_y_values.json")
    save_optimization_results(max_y_values, "max_y_values.json")

