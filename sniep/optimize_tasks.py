# optimize_tasks.py
import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, differential_evolution
from math import comb
import importlib
import logging
import time
from pathos.pools import ProcessPool as Pool

import file_utils

def load_optimization_functions(config):
    """Dynamically loads minors, jacobians, and hessians based on config['n']."""
    try:
        n = config['global_data']['n']
        module_name = f"symbolic_minors_n{n}"
        logging.info(f"Attempting to load functions from module: {module_name}")
        symbolic_module = importlib.import_module(module_name)

        funcs = {'minors': [], 'jacobians': [], 'hessians': []}
        func_types = {
            'minors': (f"calculate_S{{k}}_n{n}", lambda f: (lambda x, pf=f: -pf(x))),
            'jacobians': (f"calculate_S{{k}}_n{n}_jacobian", lambda f: (lambda x, pf=f: -np.array(pf(x)))),
            'hessians': (f"calculate_S{{k}}_n{n}_hessian", lambda f: (lambda x, pf=f: -np.array(pf(x))))
        }

        for type_key, (name_pattern, neg_wrapper) in func_types.items():
            positive_funcs = []
            for k in range(1, n + 1):
                func_name = name_pattern.format(k=k)
                if hasattr(symbolic_module, func_name):
                    positive_funcs.append(getattr(symbolic_module, func_name))
                else:
                    logging.error(f"Required function {func_name} not found in {module_name}.py")
                    raise ImportError(f"Function {func_name} not found.")

            negative_funcs = [neg_wrapper(pf) for pf in positive_funcs]
            funcs[type_key] = positive_funcs + negative_funcs # Positives first, then negatives
            logging.info(f"Successfully loaded {type_key} functions.")

        return funcs['minors'], funcs['jacobians'], funcs['hessians']

    except ImportError as e:
        logging.error(f"Failed to import module or function: {e}")
        return None, None, None
    except KeyError as e:
        logging.error(f"Configuration missing expected key 'n' or 'global_data': {e}")
        return None, None, None
    except Exception as e:
        logging.exception("An unexpected error occurred during function loading:")
        return None, None, None


def build_matrix_constraints(config):
    """Builds the linear constraints for matrix row sums."""
    try:
        n = config['global_data']['n']
        num_variables = comb(n, 2)
        A = np.zeros((n, num_variables))
        m = 0
        for j in range(n):
            for k in range(j + 1, n): # Iterate upper triangle for symmetric definition
                 A[j][m] = 1
                 A[k][m] = 1
                 m += 1
        # Constraint: sum of off-diagonal elements in each row <= 1
        # Which means sum of all elements = 1 since diagonal = 1 - sum(off-diag)
        return LinearConstraint(A, np.zeros(n), np.ones(n)) # Ensure constraints match matrix build logic
    except KeyError as e:
        logging.error(f"Config missing 'n' or 'global_data' for matrix constraints: {e}")
        return None
    except Exception as e:
        logging.exception("Error building matrix constraints:")
        return None


def run_function_with_const(loc, constraints, funcs_minors, config, func_index_for_log="unknown", run_count=0):
    """Runs differential evolution. Needs minor functions list and config."""
    try:
        n = config['global_data']['n']
        num_variables = comb(n, 2)
        # Get optimizer params from config, providing defaults
        opt_params = config.get('optimizer_params', {}) # Get section or empty dict
        g_data = config.get('global_data', {})

        tol = opt_params.get('tol', 1e-5) # Existing
        atol = opt_params.get('atol', g_data.get('atol', 1e-9)) # Check both sections
        maxiter = opt_params.get('maxiter', g_data.get('maxiter', 5000))

        popsize_k = opt_params.get('popsize_k', g_data.get('popsize_multiplier', 7))
        # Ensure popsize is at least a minimum value, e.g., 10 or 2*D
        popsize = max(10, int(popsize_k * num_variables))

        # Get other params or use scipy defaults by omitting them
        mutation = tuple(opt_params.get('mutation', (0.5, 1.0))) # Ensure tuple
        recombination = opt_params.get('recombination', 0.7)
        strategy = opt_params.get('strategy', 'best1bin')
        init_method = opt_params.get('init', g_data.get('init', 'halton')) # Allow overriding default
        attempts = opt_params.get('attempts', 5)

    except KeyError as e:
        logging.error(f"Config missing key for run_function_with_const setup: {e}")
        return None
    except Exception as e:
            logging.exception(f"Error setting up DE parameters:")
            return None


    last_result = None
    func_name = f"S{loc}" if loc < config['global_data']['n'] else f"-S{loc-config['global_data']['n']+1}"
    logging.debug(f"Starting run_function_with_const for {func_name} (loc={loc}, run={run_count})")

    for i in range(attempts):
        try:
            result = differential_evolution(
                        funcs_minors[loc],
                        bounds=[(0.0, 1.0)] * num_variables,
                        constraints=constraints,
                        polish=True,
                        init=init_method,
                        strategy=strategy,
                        maxiter=maxiter,
                        popsize=popsize,
                        atol=atol,
                        tol=tol,
                        )
            if result.success:
                logging.info(f"Optimization successful for run={run_count} on attempt {i+1}")
                logging.debug(f"Success Result:\n{result}")
                return result
            else:
                logging.warning(f"Attempt {i+1} failed for run={run_count}. Msg: {result.message}")
                logging.debug(f"Failure Result:\n{result}")
            last_result = result
        except Exception as e:
            logging.exception(f"Error in differential_evolution attempt {i+1} for {func_name}:")
            last_result = None # Reset result on error

    logging.error(f"Optimization failed after {attempts} attempts for run={run_count}. Returning last result.")
    return last_result


def optimize_func(loc, funcs_minors, funcs_jacobians, config, eqs=[], count=0):
    """Wrapper to set up constraints and call the optimizer."""
    try:
        n = config['global_data']['n']
        tol = config['global_data']['tol']
    except KeyError as e:
        logging.error(f"Config missing key for optimize_func: {e}")
        return None

    func_name = f"S{loc+1}" if loc < n else f"-S{loc-n+1}"

    constraints_list = []
    if eqs:
        for i, eq in enumerate(eqs):
            eq_func_idx, eq_target_val = eq[0], eq[1]
            if 0 <= eq_func_idx < len(funcs_jacobians):
                jac = funcs_jacobians[eq_func_idx]
                constraints_list.append(NonlinearConstraint(
                    lambda x, idx=eq_func_idx: funcs_minors[idx](x),
                    lb=eq_target_val - tol, ub=eq_target_val + tol, jac=jac
                ))
            else:
                logging.warning(f"Jacobian index {eq_func_idx} out of bounds. Constraint added w/o Jacobian.")
                constraints_list.append(NonlinearConstraint(
                    lambda x, idx=eq_func_idx: funcs_minors[idx](x),
                    lb=eq_target_val - tol, ub=eq_target_val + tol
                ))

    linear_constraints = build_matrix_constraints(config)
    if linear_constraints:
        constraints_list.append(linear_constraints)
    else:
        logging.error("Failed to build linear constraints, proceeding without them.")

    return run_function_with_const(loc, constraints_list, funcs_minors, config, func_index_for_log=func_name, run_count=count)


def build_XY_mesh(x_values, min_y_results, max_y_results, config):
    """Builds X and Y mesh points."""
    logging.info("Building XY mesh...")
    try:
        points_dim = config['global_data']['points_dim']
        num_y_points = points_dim[1] if len(points_dim) > 1 else 10
    except KeyError as e:
        logging.error(f"Config missing key for build_XY_mesh: {e}")
        return None, None
    except IndexError:
        logging.error(f"points_dim config is too short for XY mesh: {points_dim}")
        return None, None

    X_mesh, Y_mesh = [], []
    logging.debug(f"Generating {num_y_points} points in Y dim for each X.")

    for i in range(len(x_values)):
        min_y = min_y_results[i]
        max_y = max_y_results[i]
        if min_y is None or max_y is None or not hasattr(min_y, 'fun') or not hasattr(max_y, 'fun'):
             logging.warning(f"Missing valid min/max Y result at index {i} for X={x_values[i]}. Skipping.")
             continue
        min_y_val, max_y_val = min_y.fun, max_y.fun * -1 # Correct max value

        if min_y_val > max_y_val:
             logging.warning(f"Corrected Max Y ({max_y_val}) < Min Y ({min_y_val}) at X={x_values[i]}. Using bounds anyway.")

        if np.isclose(min_y_val, max_y_val): y_vals = np.full(num_y_points, min_y_val)
        elif num_y_points >= 2: y_vals = np.linspace(min_y_val, max_y_val, num_y_points)
        else: y_vals = np.array([min_y_val])

        X_mesh.append(np.full_like(y_vals, x_values[i]))
        Y_mesh.append(y_vals)

    if not X_mesh: return None, None
    final_X, final_Y = np.concatenate(X_mesh), np.concatenate(Y_mesh)
    logging.info(f"Built XY mesh with {len(final_X)} points.")
    return final_X, final_Y


def build_XYZ_mesh(X, Y, Z_min_results, Z_max_results, config):
    """Builds X, Y, and Z mesh points."""
    logging.info("Building XYZ mesh...")
    try:
        points_dim = config['global_data']['points_dim']
        num_z_points = points_dim[2] if len(points_dim) > 2 else 10
    except KeyError as e:
        logging.error(f"Config missing key for build_XYZ_mesh: {e}")
        return None, None, None
    except IndexError:
        logging.error(f"points_dim config is too short for XYZ mesh: {points_dim}")
        return None, None, None

    X_mesh, Y_mesh, Z_mesh = [], [], []
    logging.debug(f"Generating {num_z_points} points in Z dim for each (X,Y).")

    for i in range(len(X)):
        min_z = Z_min_results[i]
        max_z = Z_max_results[i]
        if min_z is None or max_z is None or not hasattr(min_z, 'fun') or not hasattr(max_z, 'fun'):
            logging.warning(f"Missing valid min/max Z result at index {i} for (X,Y)=({X[i]},{Y[i]}). Skipping.")
            continue
        min_z_val, max_z_val = min_z.fun, max_z.fun * -1 # Correct max value

        if min_z_val > max_z_val:
             logging.warning(f"Corrected Max Z ({max_z_val}) < Min Z ({min_z_val}) at (X,Y)=({X[i]},{Y[i]}). Using bounds anyway.")

        if np.isclose(min_z_val, max_z_val): z_vals = np.full(num_z_points, min_z_val)
        elif num_z_points >= 2: z_vals = np.linspace(min_z_val, max_z_val, num_z_points)
        else: z_vals = np.array([min_z_val])

        X_mesh.append(np.full_like(z_vals, X[i]))
        Y_mesh.append(np.full_like(z_vals, Y[i]))
        Z_mesh.append(z_vals)

    if not X_mesh: return None, None, None
    final_X, final_Y, final_Z = np.concatenate(X_mesh), np.concatenate(Y_mesh), np.concatenate(Z_mesh)
    logging.info(f"Built XYZ mesh with {len(final_X)} points.")
    return final_X, final_Y, final_Z


# Wrapper for parallel execution of optimize_func
def _optimize_func_parallel_wrapper(args):
    loc, funcs_minors, funcs_jacobians, config, eqs, count = args
    return optimize_func(loc, funcs_minors, funcs_jacobians, config, eqs, count)


def optimize_first_func(funcs_minors, funcs_jacobians, config, constraint_func_idx, optimize_func_idx):
    """Optimizes S_k2 constrained by S_k1, running min/max concurrently."""
    try:
        n = config['global_data']['n']
        points_dim = config['global_data']['points_dim']
        num_x_points = points_dim[0] if len(points_dim) > 0 else 10
    except KeyError as e:
        logging.error(f"Config missing key for optimize_first_func: {e}")
        return None, None, None
    except IndexError:
        logging.error(f"points_dim config is too short for optimize_first_func: {points_dim}")
        return None, None, None

    x_values = np.linspace(0, n, num_x_points)
    min_opt_loc = optimize_func_idx
    max_opt_loc = optimize_func_idx + n

    all_args = []
    for i, x_val in enumerate(x_values):
        constraints = [[constraint_func_idx, x_val]]
        all_args.append((min_opt_loc, funcs_minors, funcs_jacobians, config, constraints, i*2))
        all_args.append((max_opt_loc, funcs_minors, funcs_jacobians, config, constraints, i*2+1))

    all_results = []
    num_tasks = len(all_args)
    logging.info(f"Starting parallel optimization Stage 1 ({num_tasks} tasks: min/max pairs)...")
    try:
        with Pool() as pool:
            all_results = list(pool.imap(_optimize_func_parallel_wrapper, all_args, chunksize=1))
        logging.info(f"Finished Stage 1 parallel execution.")
    except Exception as e:
        logging.exception("Error during parallel processing in optimize_first_func:")

    # Process the combined results list, separating min and max
    min_y_results = []
    max_y_results = []
    expected_results_count = 2 * len(x_values)
    if len(all_results) == expected_results_count:
        for i in range(0, expected_results_count, 2):
            # Assuming map preserves order: result[i] is min, result[i+1] is max
            min_y_results.append(all_results[i])
            max_y_results.append(all_results[i+1])
        logging.info(f"Successfully processed {len(min_y_results)} min/max pairs for Stage 1.")
    else:
        # Handle unexpected number of results (e.g., due to errors in tasks)
        logging.error(f"Stage 1 parallel execution returned {len(all_results)} results, expected {expected_results_count}. Pairing available results.")
        num_pairs_found = len(all_results) // 2
        for i in range(0, num_pairs_found * 2, 2):
             min_y_results.append(all_results[i])
             max_y_results.append(all_results[i+1])
        if num_pairs_found < len(x_values):
             logging.warning(f"Only {num_pairs_found} complete min/max pairs were processed for Stage 1.")
        # Return potentially incomplete lists - subsequent stages need checks

    return x_values, min_y_results, max_y_results


def optimize_second_func(X, Y, funcs_minors, funcs_jacobians, config, constraint_loc_1, constraint_loc_2, func_loc):
    """Optimizes S_k3 constrained by S_k1, S_k2, running min/max concurrently."""
    try:
         n = config['global_data']['n']
    except KeyError as e:
        logging.error(f"Config missing key 'n' for optimize_second_func: {e}")
        return [], []

    if X is None or Y is None or len(X) != len(Y):
        logging.error("Invalid X or Y input provided to optimize_second_func.")
        return [], []

    total_points = len(X)
    min_opt_loc = func_loc
    max_opt_loc = func_loc + n

    # Prepare arguments for ALL tasks
    all_args = []
    for i in range(total_points):
        constraints = [[constraint_loc_1, X[i]], [constraint_loc_2, Y[i]]]
        all_args.append((min_opt_loc, funcs_minors, funcs_jacobians, config, constraints, i*2))
        all_args.append((max_opt_loc, funcs_minors, funcs_jacobians, config, constraints, i*2+1))

    all_results = []
    num_tasks = len(all_args)
    logging.info(f"Starting parallel optimization Stage 2 ({num_tasks} tasks: min/max pairs for {total_points} points)...")
    try:
        with Pool() as pool:
            all_results = list(pool.imap(_optimize_func_parallel_wrapper, all_args, chunksize=1))
        logging.info(f"Finished Stage 2 parallel execution.")
    except Exception as e:
        logging.exception("Error during parallel processing in optimize_second_func:")

    # Process the combined results list
    Z_min = []
    Z_max = []
    expected_results_count = 2 * total_points
    if len(all_results) == expected_results_count:
        for i in range(0, expected_results_count, 2):
            Z_min.append(all_results[i])
            Z_max.append(all_results[i+1])
        logging.info(f"Successfully processed {len(Z_min)} min/max pairs for Stage 2.")
    else:
        logging.error(f"Stage 2 parallel execution returned {len(all_results)} results, expected {expected_results_count}. Pairing available results.")
        num_pairs_found = len(all_results) // 2
        for i in range(0, num_pairs_found * 2, 2):
             Z_min.append(all_results[i])
             Z_max.append(all_results[i+1])
        if num_pairs_found < total_points:
            logging.warning(f"Only {num_pairs_found} complete min/max pairs were processed for Stage 2.")

    return Z_min, Z_max


def optimize_third_func(X, Y, Z, funcs_minors, funcs_jacobians, config, constraint_loc_1, constraint_loc_2, constraint_loc_3, func_loc):
    """Optimizes S_k4 constrained by S_k1, S_k2, S_k3, running min/max concurrently."""
    try:
         n = config['global_data']['n']
    except KeyError as e:
        logging.error(f"Config missing key 'n' for optimize_third_func: {e}")
        return [], []

    if X is None or Y is None or Z is None or not (len(X) == len(Y) == len(Z)):
        logging.error("Invalid X, Y, or Z input provided to optimize_third_func.")
        return [], []

    total_points = len(X)
    min_opt_loc = func_loc
    max_opt_loc = func_loc + n

    # Prepare arguments for ALL tasks
    all_args = []
    for i in range(total_points):
        constraints = [[constraint_loc_1, X[i]], [constraint_loc_2, Y[i]], [constraint_loc_3, Z[i]]]
        all_args.append((min_opt_loc, funcs_minors, funcs_jacobians, config, constraints, i*2))
        all_args.append((max_opt_loc, funcs_minors, funcs_jacobians, config, constraints, i*2+1))

    all_results = []
    num_tasks = len(all_args)
    logging.info(f"Starting parallel optimization Stage 3 ({num_tasks} tasks: min/max pairs for {total_points} points)...")
    try:
        with Pool() as pool:
            all_results = list(pool.imap(_optimize_func_parallel_wrapper, all_args, chunksize=1))
        logging.info(f"Finished Stage 3 parallel execution.")
    except Exception as e:
        logging.exception("Error during parallel processing in optimize_third_func:")

    # Process the combined results list
    W_min = []
    W_max = []
    expected_results_count = 2 * total_points
    if len(all_results) == expected_results_count:
        for i in range(0, expected_results_count, 2):
            W_min.append(all_results[i])
            W_max.append(all_results[i+1])
        logging.info(f"Successfully processed {len(W_min)} min/max pairs for Stage 3.")
    else:
        logging.error(f"Stage 3 parallel execution returned {len(all_results)} results, expected {expected_results_count}. Pairing available results.")
        num_pairs_found = len(all_results) // 2
        for i in range(0, num_pairs_found * 2, 2):
             W_min.append(all_results[i])
             W_max.append(all_results[i+1])
        if num_pairs_found < total_points:
            logging.warning(f"Only {num_pairs_found} complete min/max pairs were processed for Stage 3.")

    return W_min, W_max


def process_optimization_result(result, result_type, constraint_labels, optimized_label):
    """Helper to create the labeled dictionary from an optimization result."""
    # (Keep the implementation from the previous step)
    if result is None or not hasattr(result, 'success'):
        logging.warning(f"Skipping invalid optimization result during processing for {result_type}.")
        return None

    optimized_value = result.fun
    if result_type == "max" and optimized_value is not None:
        optimized_value *= -1

    if not hasattr(result, 'x'):
         logging.warning(f"Result object missing 'x' attribute for {result_type}. Cannot save matrix.")
         matrix_data = None
    else:
         matrix_data = result.x.tolist() if isinstance(result.x, np.ndarray) else result.x

    data_dict = {
        "type": result_type,
        **constraint_labels,
        optimized_label: optimized_value,
        "success": bool(result.success),
        "message": str(result.message),
        "matrix": matrix_data
    }
    return data_dict

def run_optimization(config):
    """Main optimization workflow called by main.py"""
    logging.info("Starting optimization process...")
    start_overall_time = time.perf_counter()

    funcs_minors, funcs_jacobians, funcs_hessians = load_optimization_functions(config)
    if funcs_minors is None:
        logging.error("Failed to load optimization functions. Aborting optimization.")
        return 1 # Indicate failure

    try:
        funcs_to_optimize = config['global_data']['funcs_to_optimize']
        if not funcs_to_optimize or len(funcs_to_optimize) < 2:
            logging.error("Config 'funcs_to_optimize' requires at least 2 functions.")
            return 1
        s_indices = [f for f in funcs_to_optimize]
    except KeyError as e:
        logging.error(f"Config missing key for optimization setup: {e}")
        return 1

    all_results = []
    last_stage_success = False

    # --- Stage 1 ---
    constraint1_idx = funcs_to_optimize[0]
    optimize1_idx = funcs_to_optimize[1]
    label_constraint1 = f"S{s_indices[0]}_constraint"
    label_optimize1 = f"S{s_indices[1]}_optimized"
    logging.info(f"--- Running Stage 1 ({label_optimize1} constrained by {label_constraint1}) ---")
    stage1_start = time.perf_counter()
    x_values, min_y_results, max_y_results = optimize_first_func(
        funcs_minors, funcs_jacobians, config, constraint1_idx, optimize1_idx
    )
    logging.info(f"--- Stage 1 finished in {time.perf_counter() - stage1_start:.4f} seconds ---")

    combined_results_stage1 = []
    if x_values is not None: # Check if optimization stage itself failed critically
        for i in range(len(x_values)):
            constraints = {label_constraint1: x_values[i]}
            min_dict = process_optimization_result(min_y_results[i], "min", constraints, label_optimize1)
            max_dict = process_optimization_result(max_y_results[i], "max", constraints, label_optimize1)
            if min_dict: combined_results_stage1.append(min_dict)
            if max_dict: combined_results_stage1.append(max_dict)

    if not combined_results_stage1:
        logging.error("Stage 1 yielded no valid results.")
        return 1
    all_results = combined_results_stage1
    last_stage_success = True # Mark success to potentially save later if subsequent stages fail

    # --- Stage 2 ---
    if len(funcs_to_optimize) >= 3:
        last_stage_success = False # Reset for this stage
        constraint2_idx = funcs_to_optimize[1]
        optimize2_idx = funcs_to_optimize[2]
        label_constraint2 = f"S{s_indices[1]}_constraint"
        label_optimize2 = f"S{s_indices[2]}_optimized"
        logging.info(f"--- Running Stage 2 ({label_optimize2} constrained by {label_constraint1}, {label_constraint2}) ---")
        stage2_start = time.perf_counter()
        X_mesh, Y_mesh = build_XY_mesh(x_values, min_y_results, max_y_results, config)

        if X_mesh is None:
            logging.error("Failed to build XY mesh for Stage 2.")
        else:
            min_z_results, max_z_results = optimize_second_func(
                X_mesh, Y_mesh, funcs_minors, funcs_jacobians, config,
                constraint1_idx, constraint2_idx, optimize2_idx
            )
            logging.info(f"--- Stage 2 finished in {time.perf_counter() - stage2_start:.4f} seconds ---")

            combined_results_stage2 = []
            for i in range(len(X_mesh)):
                constraints = {label_constraint1: X_mesh[i], label_constraint2: Y_mesh[i]}
                min_dict = process_optimization_result(min_z_results[i], "min", constraints, label_optimize2)
                max_dict = process_optimization_result(max_z_results[i], "max", constraints, label_optimize2)
                if min_dict: combined_results_stage2.append(min_dict)
                if max_dict: combined_results_stage2.append(max_dict)

            if combined_results_stage2:
                all_results = combined_results_stage2
                last_stage_success = True
            else:
                 logging.error("Stage 2 yielded no valid results.")
                 # Keep Stage 1 results in all_results

    # --- Stage 3 ---
    if len(funcs_to_optimize) >= 4:
         last_stage_success = False # Reset
         constraint3_idx = funcs_to_optimize[2]
         optimize3_idx = funcs_to_optimize[3]
         label_constraint3 = f"S{s_indices[2]}_constraint"
         label_optimize3 = f"S{s_indices[3]}_optimized"
         logging.info(f"--- Running Stage 3 ({label_optimize3} constrained by {label_constraint1}, {label_constraint2}, {label_constraint3}) ---")
         stage3_start = time.perf_counter()
         # Need results from previous stage (potentially failed)
         # Need to handle the case where X_mesh/Y_mesh were not built or min_z/max_z were empty
         if 'X_mesh' in locals() and X_mesh is not None:
             X_mesh_3d, Y_mesh_3d, Z_mesh_3d = build_XYZ_mesh(X_mesh, Y_mesh, min_z_results, max_z_results, config)
             if X_mesh_3d is None:
                 logging.error("Failed to build XYZ mesh for Stage 3.")
             else:
                min_w_results, max_w_results = optimize_third_func(
                    X_mesh_3d, Y_mesh_3d, Z_mesh_3d, funcs_minors, funcs_jacobians, config,
                    constraint1_idx, constraint2_idx, constraint3_idx, optimize3_idx
                )
                logging.info(f"--- Stage 3 finished in {time.perf_counter() - stage3_start:.4f} seconds ---")

                combined_results_stage3 = []
                for i in range(len(X_mesh_3d)):
                    constraints = {
                         label_constraint1: X_mesh_3d[i], label_constraint2: Y_mesh_3d[i], label_constraint3: Z_mesh_3d[i]
                    }
                    min_dict = process_optimization_result(min_w_results[i], "min", constraints, label_optimize3)
                    max_dict = process_optimization_result(max_w_results[i], "max", constraints, label_optimize3)
                    if min_dict: combined_results_stage3.append(min_dict)
                    if max_dict: combined_results_stage3.append(max_dict)

                if combined_results_stage3:
                     all_results = combined_results_stage3
                     last_stage_success = True
                else:
                     logging.error("Stage 3 yielded no valid results.")
                     # Keep results from previous successful stage in all_results
         else:
              logging.error("Skipping Stage 3 due to missing inputs from previous stage.")


    # --- Save Results ---
    if all_results and last_stage_success:
        output_filename = file_utils.build_file_name(config, is_coef=True)
        save_success = file_utils.save_results(all_results, output_filename)
        if save_success:
             logging.info(f"Optimization results saved successfully. Total time: {time.perf_counter() - start_overall_time:.4f} seconds.")
             return 0 # Success
        else:
             logging.error("Failed to save optimization results.")
             return 1 # Failure
    elif all_results and not last_stage_success:
         logging.warning("Saving results from the last successful stage.")
         output_filename = file_utils.build_file_name(config, is_coef=True)
         save_success = file_utils.save_results(all_results, output_filename)
         return 1 # Indicate partial failure / last stage failed
    else:
        logging.error("No optimization results were generated or the process failed critically.")
        return 1 # Failure