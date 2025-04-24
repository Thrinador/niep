import numpy as np
from scipy.optimize import (
    LinearConstraint,
    NonlinearConstraint,
    minimize,
    Bounds,
)
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
                    if type_key == 'hessians':
                         logging.warning(f"Optional Hessian function {func_name} not found in {module_name}.py")
                    else:
                         logging.error(f"Required function {func_name} not found in {module_name}.py")
                         raise ImportError(f"Function {func_name} not found.")

            # Apply negation wrapper only if positive functions were found
            negative_funcs = [neg_wrapper(pf) for pf in positive_funcs if pf is not None] if positive_funcs else []

            # Store positive funcs first, then negatives
            # Handle case where hessians might be missing
            if type_key == 'hessians' and not any(positive_funcs):
                 # Fill hessians list with None placeholders if none were found
                 num_expected = 2 * n
                 funcs[type_key] = [None] * num_expected
                 logging.warning(f"No Hessian functions loaded for n={n}. 'trust-constr' might fall back or fail if Hessian is required.")
            else:
                 funcs[type_key] = positive_funcs + negative_funcs

            logging.info(f"Successfully loaded {len(positive_funcs)} positive and {len(negative_funcs)} negative {type_key} functions.")


        # Final check: Ensure all lists have the expected length (2n) even if functions are None
        num_expected = 2 * n
        for type_key in funcs:
            if len(funcs[type_key]) != num_expected:
                logging.warning(f"Loaded {len(funcs[type_key])} {type_key} functions, expected {num_expected}. Check symbolic module.")
                # Pad with None if needed, although load logic should handle this
                while len(funcs[type_key]) < num_expected:
                    funcs[type_key].append(None)


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
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                A[i, idx] = 1
                A[j, idx] = 1
                idx += 1

        # Constraint: sum of off-diagonal elements in each row <= 1
        # Since diagonal P_ii = 1 - sum(P_ij for j!=i), this ensures P_ii >= 0.
        # The upper bound ensures row sums don't exceed 1. Variables bounded >= 0.
        # lb=0 ensures row sums are non-negative (implicit via variable bounds)
        # ub=1 ensures row sums are <= 1
        return LinearConstraint(A, np.zeros(n), np.ones(n))
    except KeyError as e:
        logging.error(f"Config missing 'n' or 'global_data' for matrix constraints: {e}")
        return None
    except Exception as e:
        logging.exception("Error building matrix constraints:")
        return None

def run_function_with_const(
    loc,
    constraints,
    funcs_minors,
    funcs_jacobians,
    config,
    func_index_for_log="unknown",
    run_count=0
):
    """
    Runs scipy.optimize.minimize with 'SLSQP', attempting multiple
    random starting points if the first fails.
    """
    try:
        n = config['global_data']['n']
        num_variables = comb(n, 2)
        opt_params = config.get('optimizer_params', {})
        g_data = config.get('global_data', {})

        tol = opt_params.get('tol', g_data.get('tol', 1e-6))
        maxiter = opt_params.get('maxiter', g_data.get('maxiter', 100))
        attempts = opt_params.get('attempts', g_data.get('attempts', 5))

        logging.debug(
            f"Minimize ('SLSQP') Params: ftol={tol}, maxiter={maxiter}, attempts={attempts}"
        )

    except KeyError as e:
        logging.error(f"Config missing key for run_function_with_const setup: {e}")
        return None
    except Exception as e:
        logging.exception(f"Error setting up minimize parameters:")
        return None

    # --- Prepare unchanging parts for minimize ---
    obj_func = funcs_minors[loc]
    bounds = Bounds([0.0] * num_variables, [1.0] * num_variables)
    jac = funcs_jacobians[loc] if funcs_jacobians and loc < len(funcs_jacobians) else None

    if jac is None:
         logging.error(f"Jacobian for objective function index {loc} not available for run {run_count}. SLSQP cannot proceed.")
         return None

    func_name = f"S{loc+1}" if loc < n else f"-S{loc-n+1}"

    last_result = None
    for i in range(attempts):
        x0 = np.random.rand(num_variables)

        try:
            result = minimize(
                obj_func,
                x0,
                method='SLSQP',
                jac=jac,
                constraints=constraints, 
                bounds=bounds,
                options={
                    'maxiter': maxiter,
                    'ftol': tol,
                    'disp': False
                    }
            )
            last_result = result

            if result.success:
                if run_count % 103 == 0: # Reduce the number of lines printed on larger runs
                    logging.info(f"Optimization successful for {func_name} run={run_count} on attempt {i+1}.")
                logging.debug(f"Success Result:\n{result}")
                return result
            # else:
                # SLSQP failure messages can be informative
                # logging.debug(f"Failure Result (Attempt {i+1}):\n{result}")

        except (ValueError, TypeError) as e:
             logging.exception(f"Error during minimize attempt {i+1} for {func_name} run={run_count}: {e}. Check function/jacobian.")
             last_result = None
        except Exception as e:
             logging.exception(f"Unexpected Error in minimize attempt {i+1} for {func_name} run={run_count}:")
             last_result = None

    logging.error(f"Optimization failed for {func_name} run={run_count} after {attempts} attempts.")
    if last_result is None:
        logging.error(f"All attempts failed critically (e.g., function evaluation errors) for {func_name} run={run_count}.")
    return last_result


def optimize_func(loc, funcs_minors, funcs_jacobians, funcs_hessians, config, eqs=[], count=0):
    """Wrapper to set up constraints and call the optimizer."""
    try:
        n = config['global_data']['n']
        tol = config['global_data']['tol']
    except KeyError as e:
        logging.error(f"Config missing key for optimize_func: {e}")
        return None

    func_name = f"S{loc+1}" if loc < n else f"-S{loc-n+1}"

    constraints_list = []
    # --- Nonlinear constraints from 'eqs' argument ---
    if eqs:
        for i, eq in enumerate(eqs):
            eq_func_idx, eq_target_val = eq[0], eq[1]
            if 0 <= eq_func_idx < len(funcs_minors):
                nlc_func = lambda x, idx=eq_func_idx: funcs_minors[idx](x)
                nlc_jac = funcs_jacobians[eq_func_idx] if funcs_jacobians and 0 <= eq_func_idx < len(funcs_jacobians) else '2-point'

                constraints_list.append(NonlinearConstraint(
                    nlc_func,
                    lb=eq_target_val - tol,
                    ub=eq_target_val + tol,
                    jac=nlc_jac
                ))
            else:
                 logging.error(f"Constraint function index {eq_func_idx} is out of bounds. Skipping constraint.")

    # --- Linear constraints ---
    linear_constraints = build_matrix_constraints(config)
    if linear_constraints:
        constraints_list.append(linear_constraints)
    else:
        logging.error("Failed to build linear matrix constraints for run {count}! Optimization might fail or give invalid results.")

    return run_function_with_const(
        loc,
        constraints_list,
        funcs_minors,
        funcs_jacobians,
        config,
        func_index_for_log=func_name,
        run_count=count
    )


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
        # Check if results are valid OptimizeResult objects and have 'fun'
        if min_y is None or max_y is None or not hasattr(min_y, 'fun') or not hasattr(max_y, 'fun') or min_y.fun is None or max_y.fun is None:
             logging.warning(f"Missing valid min/max Y result data at index {i} for X={x_values[i]}. Skipping.")
             continue
        min_y_val, max_y_val = min_y.fun, max_y.fun * -1 # Correct max value (was minimized negative)

        # Ensure min <= max after potential floating point inaccuracies
        if min_y_val > max_y_val:
            if np.isclose(min_y_val, max_y_val):
                 min_y_val = max_y_val # Force equality if close
            else:
                 logging.warning(f"Corrected Max Y ({max_y_val:.4f}) < Min Y ({min_y_val:.4f}) at X={x_values[i]:.4f}. Using bounds anyway.")

        if np.isclose(min_y_val, max_y_val): y_vals = np.full(num_y_points, min_y_val)
        elif num_y_points >= 2: y_vals = np.linspace(min_y_val, max_y_val, num_y_points)
        elif num_y_points == 1: y_vals = np.array([(min_y_val + max_y_val) / 2.0]) # Midpoint if 1 point requested
        else: y_vals = np.array([]) # No points if num_y_points is 0 or less


        if y_vals.size > 0:
             X_mesh.append(np.full_like(y_vals, x_values[i]))
             Y_mesh.append(y_vals)

    if not X_mesh:
         logging.warning("XY Mesh generation yielded no points.")
         return None, None
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
        # Check if results are valid OptimizeResult objects and have 'fun'
        if min_z is None or max_z is None or not hasattr(min_z, 'fun') or not hasattr(max_z, 'fun') or min_z.fun is None or max_z.fun is None:
            logging.warning(f"Missing valid min/max Z result data at index {i} for (X,Y)=({X[i]:.4f},{Y[i]:.4f}). Skipping.")
            continue
        min_z_val, max_z_val = min_z.fun, max_z.fun * -1 # Correct max value

        # Ensure min <= max after potential floating point inaccuracies
        if min_z_val > max_z_val:
             if np.isclose(min_z_val, max_z_val):
                 min_z_val = max_z_val # Force equality if close
             else:
                 logging.warning(f"Corrected Max Z ({max_z_val:.4f}) < Min Z ({min_z_val:.4f}) at (X,Y)=({X[i]:.4f},{Y[i]:.4f}). Using bounds anyway.")
                 # Optional: Swap them if strict ordering is needed downstream
                 # min_z_val, max_z_val = max_z_val, min_z_val

        if np.isclose(min_z_val, max_z_val): z_vals = np.full(num_z_points, min_z_val)
        elif num_z_points >= 2: z_vals = np.linspace(min_z_val, max_z_val, num_z_points)
        elif num_z_points == 1: z_vals = np.array([(min_z_val + max_z_val) / 2.0]) # Midpoint if 1 point requested
        else: z_vals = np.array([]) # No points if num_z_points is 0 or less

        if z_vals.size > 0:
             X_mesh.append(np.full_like(z_vals, X[i]))
             Y_mesh.append(np.full_like(z_vals, Y[i]))
             Z_mesh.append(z_vals)

    if not X_mesh:
         logging.warning("XYZ Mesh generation yielded no points.")
         return None, None, None
    final_X, final_Y, final_Z = np.concatenate(X_mesh), np.concatenate(Y_mesh), np.concatenate(Z_mesh)
    logging.info(f"Built XYZ mesh with {len(final_X)} points.")
    return final_X, final_Y, final_Z


# Modified Wrapper for parallel execution of optimize_func
def _optimize_func_parallel_wrapper(args):
    # Unpack hessians as well
    loc, funcs_minors, funcs_jacobians, funcs_hessians, config, eqs, count = args
    # Pass hessians to optimize_func
    return optimize_func(loc, funcs_minors, funcs_jacobians, funcs_hessians, config, eqs, count)

# Modified to include hessians in args
def optimize_first_func(funcs_minors, funcs_jacobians, funcs_hessians, config, constraint_func_idx, optimize_func_idx):
    """Optimizes S_k2 constrained by S_k1, running min/max concurrently."""
    try:
        n = config['global_data']['n']
        points_dim = config['global_data']['points_dim']
        num_x_points = points_dim[0] if len(points_dim) > 0 else 10
        num_workers = config['global_data'].get('num_workers', None) # Get num_workers for Pool
    except KeyError as e:
        logging.error(f"Config missing key for optimize_first_func: {e}")
        return None, None, None
    except IndexError:
        logging.error(f"points_dim config is too short for optimize_first_func: {points_dim}")
        return None, None, None

    # Ensure num_x_points is at least 2 for linspace to work as expected unless explicitly 1
    if num_x_points < 1:
        logging.warning(f"num_x_points ({num_x_points}) is less than 1, setting to 1.")
        num_x_points = 1
    elif num_x_points == 1:
         x_values = np.array([config['global_data'].get('single_x_value', n / 2.0)]) # Default to midpoint if only 1 point
         logging.info(f"Running Stage 1 for single X value: {x_values[0]:.4f}")
    else:
        x_min = config['global_data'].get('x_min', 0.0) # Allow specifying range
        x_max = config['global_data'].get('x_max', float(n))
        x_values = np.linspace(x_min, x_max, num_x_points)
        logging.info(f"Running Stage 1 for {num_x_points} X values from {x_min:.4f} to {x_max:.4f}")


    min_opt_loc = optimize_func_idx
    max_opt_loc = optimize_func_idx + n

    # Pack arguments including funcs_hessians
    all_args = []
    for i, x_val in enumerate(x_values):
        constraints = [[constraint_func_idx, x_val]]
        # Args: loc, funcs_minors, funcs_jacobians, funcs_hessians, config, eqs, count
        all_args.append((min_opt_loc, funcs_minors, funcs_jacobians, funcs_hessians, config, constraints, i*2))
        all_args.append((max_opt_loc, funcs_minors, funcs_jacobians, funcs_hessians, config, constraints, i*2+1))

    all_results = []
    num_tasks = len(all_args)
    logging.info(f"Starting parallel optimization Stage 1 ({num_tasks} tasks: min/max pairs for {len(x_values)} X values)... using {num_workers or 'default'} workers.")
    start_time = time.perf_counter()
    try:
        # Use specified number of workers if provided
        with Pool(nodes=num_workers) as pool:
            # Consider adjusting chunksize based on task duration vs overhead
            all_results = list(pool.imap(_optimize_func_parallel_wrapper, all_args, chunksize=1))
        end_time = time.perf_counter()
        logging.info(f"Finished Stage 1 parallel execution in {end_time - start_time:.4f} seconds.")
    except Exception as e:
        logging.exception("Error during parallel processing in optimize_first_func:")
        # Attempt to salvage any results obtained before the error
        # This might depend on the specifics of pathos/Pool behavior on error

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
        # Try to pair results robustly, assuming they might be interleaved if order isn't guaranteed or tasks failed
        temp_results = {arg[-1]: res for arg, res in zip(all_args, all_results) if res is not None} # Map count to result
        for i in range(len(x_values)):
             min_res = temp_results.get(i*2)
             max_res = temp_results.get(i*2+1)
             min_y_results.append(min_res) # Append None if missing
             max_y_results.append(max_res) # Append None if missing

        num_pairs_found = sum(1 for min_r, max_r in zip(min_y_results, max_y_results) if min_r is not None and max_r is not None)
        num_total_processed = sum(1 for r in min_y_results + max_y_results if r is not None)
        logging.warning(f"Processed {num_total_processed} individual results, forming {num_pairs_found} complete min/max pairs for Stage 1.")


    return x_values, min_y_results, max_y_results

# Modified to include hessians in args
def optimize_second_func(X, Y, funcs_minors, funcs_jacobians, funcs_hessians, config, constraint_loc_1, constraint_loc_2, func_loc):
    """Optimizes S_k3 constrained by S_k1, S_k2, running min/max concurrently."""
    try:
         n = config['global_data']['n']
         num_workers = config['global_data'].get('num_workers', None) # Get num_workers for Pool
    except KeyError as e:
        logging.error(f"Config missing key for optimize_second_func: {e}")
        return [], []

    if X is None or Y is None or len(X) != len(Y):
        logging.error("Invalid X or Y input provided to optimize_second_func. Aborting Stage 2.")
        return [], []
    if len(X) == 0:
        logging.warning("X and Y inputs for Stage 2 are empty. No optimization tasks to run.")
        return [], []


    total_points = len(X)
    min_opt_loc = func_loc
    max_opt_loc = func_loc + n

    # Prepare arguments for ALL tasks, including hessians
    all_args = []
    for i in range(total_points):
        constraints = [[constraint_loc_1, X[i]], [constraint_loc_2, Y[i]]]
        # Args: loc, funcs_minors, funcs_jacobians, funcs_hessians, config, eqs, count
        all_args.append((min_opt_loc, funcs_minors, funcs_jacobians, funcs_hessians, config, constraints, i*2))
        all_args.append((max_opt_loc, funcs_minors, funcs_jacobians, funcs_hessians, config, constraints, i*2+1))

    all_results = []
    num_tasks = len(all_args)
    logging.info(f"Starting parallel optimization Stage 2 ({num_tasks} tasks: min/max pairs for {total_points} (X,Y) points)... using {num_workers or 'default'} workers.")
    start_time = time.perf_counter()
    try:
        with Pool(nodes=num_workers) as pool:
            all_results = list(pool.imap(_optimize_func_parallel_wrapper, all_args, chunksize=1))
        end_time = time.perf_counter()
        logging.info(f"Finished Stage 2 parallel execution in {end_time - start_time:.4f} seconds.")
    except Exception as e:
        logging.exception("Error during parallel processing in optimize_second_func:")


    # Process the combined results list, separating min and max
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
        # Robust pairing
        temp_results = {arg[-1]: res for arg, res in zip(all_args, all_results) if res is not None} # Map count to result
        for i in range(total_points):
             min_res = temp_results.get(i*2)
             max_res = temp_results.get(i*2+1)
             Z_min.append(min_res) # Append None if missing
             Z_max.append(max_res) # Append None if missing

        num_pairs_found = sum(1 for min_r, max_r in zip(Z_min, Z_max) if min_r is not None and max_r is not None)
        num_total_processed = sum(1 for r in Z_min + Z_max if r is not None)
        logging.warning(f"Processed {num_total_processed} individual results, forming {num_pairs_found} complete min/max pairs for Stage 2.")


    return Z_min, Z_max

# Modified to include hessians in args
def optimize_third_func(X, Y, Z, funcs_minors, funcs_jacobians, funcs_hessians, config, constraint_loc_1, constraint_loc_2, constraint_loc_3, func_loc):
    """Optimizes S_k4 constrained by S_k1, S_k2, S_k3, running min/max concurrently."""
    try:
         n = config['global_data']['n']
         num_workers = config['global_data'].get('num_workers', None) # Get num_workers for Pool
    except KeyError as e:
        logging.error(f"Config missing key for optimize_third_func: {e}")
        return [], []

    if X is None or Y is None or Z is None or not (len(X) == len(Y) == len(Z)):
        logging.error("Invalid X, Y, or Z input provided to optimize_third_func. Aborting Stage 3.")
        return [], []
    if len(X) == 0:
        logging.warning("X, Y, Z inputs for Stage 3 are empty. No optimization tasks to run.")
        return [], []

    total_points = len(X)
    min_opt_loc = func_loc
    max_opt_loc = func_loc + n

    all_args = []
    for i in range(total_points):
        constraints = [[constraint_loc_1, X[i]], [constraint_loc_2, Y[i]], [constraint_loc_3, Z[i]]]
        all_args.append((min_opt_loc, funcs_minors, funcs_jacobians, funcs_hessians, config, constraints, i*2))
        all_args.append((max_opt_loc, funcs_minors, funcs_jacobians, funcs_hessians, config, constraints, i*2+1))

    all_results = []
    num_tasks = len(all_args)
    logging.info(f"Starting parallel optimization Stage 3 ({num_tasks} tasks: min/max pairs for {total_points} (X,Y,Z) points)... using {num_workers or 'default'} workers.")
    start_time = time.perf_counter()
    try:
        with Pool(nodes=num_workers) as pool:
            all_results = list(pool.imap(_optimize_func_parallel_wrapper, all_args, chunksize=1))
        end_time = time.perf_counter()
        logging.info(f"Finished Stage 3 parallel execution in {end_time - start_time:.4f} seconds.")
    except Exception as e:
        logging.exception("Error during parallel processing in optimize_third_func:")

    # Process the combined results list, separating min and max
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
         # Robust pairing
        temp_results = {arg[-1]: res for arg, res in zip(all_args, all_results) if res is not None} # Map count to result
        for i in range(total_points):
             min_res = temp_results.get(i*2)
             max_res = temp_results.get(i*2+1)
             W_min.append(min_res) # Append None if missing
             W_max.append(max_res) # Append None if missing

        num_pairs_found = sum(1 for min_r, max_r in zip(W_min, W_max) if min_r is not None and max_r is not None)
        num_total_processed = sum(1 for r in W_min + W_max if r is not None)
        logging.warning(f"Processed {num_total_processed} individual results, forming {num_pairs_found} complete min/max pairs for Stage 3.")


    return W_min, W_max


def process_optimization_result(result, result_type, constraint_labels, optimized_label):
    """Helper to create the labeled dictionary from an optimization result."""
    # Check if result is None or lacks necessary attributes
    if result is None or not hasattr(result, 'success') or not hasattr(result, 'fun') or result.fun is None:
        logging.warning(f"Skipping invalid/incomplete optimization result during processing for {result_type} {optimized_label} with constraints {constraint_labels}.")
        return None

    optimized_value = result.fun
    # Correct max value (we minimize the negative function for maximization)
    if result_type == "max":
        optimized_value *= -1

    # Check for solution vector 'x'
    if not hasattr(result, 'x') or result.x is None:
         logging.warning(f"Result object missing valid 'x' attribute for {result_type} {optimized_label}. Cannot save matrix.")
         matrix_data = None
    else:
         # Ensure 'x' is converted to a list for JSON serialization
         matrix_data = result.x.tolist() if isinstance(result.x, np.ndarray) else list(result.x)

    # Get message, ensuring it's a string
    message = str(getattr(result, 'message', 'No message provided'))


    data_dict = {
        "type": result_type,
        **constraint_labels, # Unpack constraint key-value pairs
        optimized_label: optimized_value,
        "success": bool(result.success), # Ensure boolean
        "message": message,
        "matrix": matrix_data # Can be None if 'x' was missing
    }
    return data_dict

# Main workflow modified to pass hessians
def run_optimization(config):
    """Main optimization workflow called by main.py"""
    logging.info("Starting optimization process...")
    start_overall_time = time.perf_counter()

    funcs_minors, funcs_jacobians, funcs_hessians = load_optimization_functions(config)
    # Check if essential functions (minors, jacobians) loaded
    if funcs_minors is None or funcs_jacobians is None:
        logging.error("Failed to load essential optimization functions (minors/jacobians). Aborting optimization.")
        return 1 # Indicate failure
    if funcs_hessians is None:
         logging.warning("Hessian functions failed to load. Optimizer will approximate.")
         # Proceed, but trust-constr will approximate Hessians


    try:
        funcs_to_optimize = config['global_data']['funcs_to_optimize']
        if not isinstance(funcs_to_optimize, list) or len(funcs_to_optimize) < 2:
            logging.error("Config 'funcs_to_optimize' must be a list with at least 2 function indices (e.g., [1, 2]).")
            return 1
        # Convert 1-based Sk index from config to 0-based list index
        s_indices = [f-1 for f in funcs_to_optimize] # Store original 1-based for labels
        constraint_indices = [f-1 for f in funcs_to_optimize] # 0-based for list access
        logging.info(f"Optimization plan: Constraint order S{funcs_to_optimize}, Indices {constraint_indices}")
    except KeyError as e:
        logging.error(f"Config missing key for optimization setup: {e}")
        return 1
    except Exception as e:
         logging.exception(f"Error processing 'funcs_to_optimize' from config:")
         return 1


    all_results_dicts = []
    last_stage_success = False
    x_values_stage1, y_min_results_stage1, y_max_results_stage1 = None, None, None
    x_mesh_stage2, y_mesh_stage2 = None, None
    z_min_results_stage2, z_max_results_stage2 = None, None
    x_mesh_stage3, y_mesh_stage3, z_mesh_stage3 = None, None, None
    w_min_results_stage3, w_max_results_stage3 = None, None


    # --- Stage 1 ---
    constraint1_idx = constraint_indices[0]
    optimize1_idx = constraint_indices[1]
    label_constraint1 = f"S{funcs_to_optimize[0]}_constraint"
    label_optimize1 = f"S{funcs_to_optimize[1]}_optimized"
    logging.info(f"--- Running Stage 1 ({label_optimize1} constrained by {label_constraint1}) ---")
    stage1_start = time.perf_counter()
    x_values_stage1, y_min_results_stage1, y_max_results_stage1 = optimize_first_func(
        funcs_minors, funcs_jacobians, funcs_hessians, config, constraint1_idx, optimize1_idx
    )
    logging.info(f"--- Stage 1 finished in {time.perf_counter() - stage1_start:.4f} seconds ---")

    current_stage_results = []
    if x_values_stage1 is not None and y_min_results_stage1 is not None and y_max_results_stage1 is not None:
        # Check lengths match before processing
        if not (len(x_values_stage1) == len(y_min_results_stage1) == len(y_max_results_stage1)):
             logging.error(f"Mismatched result lengths in Stage 1: X({len(x_values_stage1)}), Ymin({len(y_min_results_stage1)}), Ymax({len(y_max_results_stage1)})")
        else:
             for i in range(len(x_values_stage1)):
                 constraints = {label_constraint1: x_values_stage1[i]}
                 min_dict = process_optimization_result(y_min_results_stage1[i], "min", constraints, label_optimize1)
                 max_dict = process_optimization_result(y_max_results_stage1[i], "max", constraints, label_optimize1)
                 if min_dict: current_stage_results.append(min_dict)
                 if max_dict: current_stage_results.append(max_dict)

    if not current_stage_results:
        logging.error("Stage 1 yielded no valid processed results. Aborting.")
        return 1 # Critical failure if first stage yields nothing
    all_results_dicts = current_stage_results
    last_stage_success = True # Mark success


    # --- Stage 2 ---
    if len(funcs_to_optimize) >= 3:
        last_stage_success = False # Reset for this stage
        constraint2_idx = constraint_indices[1]
        optimize2_idx = constraint_indices[2]
        label_constraint2 = f"S{funcs_to_optimize[1]}_constraint"
        label_optimize2 = f"S{funcs_to_optimize[2]}_optimized"
        logging.info(f"--- Running Stage 2 ({label_optimize2} constrained by {label_constraint1}, {label_constraint2}) ---")
        stage2_start = time.perf_counter()
        # Build mesh based on Stage 1 results
        x_mesh_stage2, y_mesh_stage2 = build_XY_mesh(x_values_stage1, y_min_results_stage1, y_max_results_stage1, config)

        if x_mesh_stage2 is None or y_mesh_stage2 is None:
            logging.error("Failed to build XY mesh for Stage 2. Cannot proceed with Stage 2.")
        else:
            z_min_results_stage2, z_max_results_stage2 = optimize_second_func(
                x_mesh_stage2, y_mesh_stage2, funcs_minors, funcs_jacobians, funcs_hessians, config,
                constraint1_idx, constraint2_idx, optimize2_idx
            )
            logging.info(f"--- Stage 2 finished in {time.perf_counter() - stage2_start:.4f} seconds ---")

            current_stage_results = []
            if z_min_results_stage2 is not None and z_max_results_stage2 is not None:
                 # Check lengths match
                 if not (len(x_mesh_stage2) == len(y_mesh_stage2) == len(z_min_results_stage2) == len(z_max_results_stage2)):
                       logging.error(f"Mismatched result lengths in Stage 2 processing: XY({len(x_mesh_stage2)}), Zmin({len(z_min_results_stage2)}), Zmax({len(z_max_results_stage2)})")
                 else:
                     for i in range(len(x_mesh_stage2)):
                         constraints = {label_constraint1: x_mesh_stage2[i], label_constraint2: y_mesh_stage2[i]}
                         min_dict = process_optimization_result(z_min_results_stage2[i], "min", constraints, label_optimize2)
                         max_dict = process_optimization_result(z_max_results_stage2[i], "max", constraints, label_optimize2)
                         if min_dict: current_stage_results.append(min_dict)
                         if max_dict: current_stage_results.append(max_dict)

            if current_stage_results:
                all_results_dicts = current_stage_results # Replace with Stage 2 results
                last_stage_success = True
            else:
                 logging.error("Stage 2 yielded no valid processed results.")
                 # Keep Stage 1 results in all_results_dicts if Stage 2 fails


    # --- Stage 3 ---
    if len(funcs_to_optimize) >= 4:
         last_stage_success = False # Reset
         constraint3_idx = constraint_indices[2]
         optimize3_idx = constraint_indices[3]
         label_constraint3 = f"S{funcs_to_optimize[2]}_constraint"
         label_optimize3 = f"S{funcs_to_optimize[3]}_optimized"
         logging.info(f"--- Running Stage 3 ({label_optimize3} constrained by {label_constraint1}, {label_constraint2}, {label_constraint3}) ---")
         stage3_start = time.perf_counter()

         # Check if inputs from Stage 2 are available
         if x_mesh_stage2 is None or y_mesh_stage2 is None or z_min_results_stage2 is None or z_max_results_stage2 is None:
              logging.error("Skipping Stage 3 due to missing or failed inputs from Stage 2.")
         else:
             x_mesh_stage3, y_mesh_stage3, z_mesh_stage3 = build_XYZ_mesh(
                 x_mesh_stage2, y_mesh_stage2, z_min_results_stage2, z_max_results_stage2, config
                 )
             if x_mesh_stage3 is None or y_mesh_stage3 is None or z_mesh_stage3 is None:
                 logging.error("Failed to build XYZ mesh for Stage 3. Cannot proceed.")
             else:
                w_min_results_stage3, w_max_results_stage3 = optimize_third_func(
                    x_mesh_stage3, y_mesh_stage3, z_mesh_stage3, funcs_minors, funcs_jacobians, funcs_hessians, config,
                    constraint1_idx, constraint2_idx, constraint3_idx, optimize3_idx
                )
                logging.info(f"--- Stage 3 finished in {time.perf_counter() - stage3_start:.4f} seconds ---")

                current_stage_results = []
                if w_min_results_stage3 is not None and w_max_results_stage3 is not None:
                    # Check lengths match
                     if not (len(x_mesh_stage3) == len(y_mesh_stage3) == len(z_mesh_stage3) == len(w_min_results_stage3) == len(w_max_results_stage3)):
                         logging.error(f"Mismatched result lengths in Stage 3 processing: XYZ({len(x_mesh_stage3)}), Wmin({len(w_min_results_stage3)}), Wmax({len(w_max_results_stage3)})")
                     else:
                         for i in range(len(x_mesh_stage3)):
                             constraints = {
                                 label_constraint1: x_mesh_stage3[i],
                                 label_constraint2: y_mesh_stage3[i],
                                 label_constraint3: z_mesh_stage3[i]
                             }
                             min_dict = process_optimization_result(w_min_results_stage3[i], "min", constraints, label_optimize3)
                             max_dict = process_optimization_result(w_max_results_stage3[i], "max", constraints, label_optimize3)
                             if min_dict: current_stage_results.append(min_dict)
                             if max_dict: current_stage_results.append(max_dict)

                if current_stage_results:
                     all_results_dicts = current_stage_results # Replace with Stage 3 results
                     last_stage_success = True
                else:
                     logging.error("Stage 3 yielded no valid processed results.")
                     # Keep results from previous successful stage


    # --- Save Results ---
    if all_results_dicts: # Save if any results were generated
        output_filename = file_utils.build_file_name(config, is_coef=True) # Assumes build_file_name exists
        save_success = file_utils.save_results(all_results_dicts, output_filename) # Assumes save_results exists

        total_time = time.perf_counter() - start_overall_time
        if save_success:
            if last_stage_success:
                 logging.info(f"Optimization completed and results saved successfully to {output_filename}. Total time: {total_time:.4f} seconds.")
                 return 0 # Indicate full success
            else:
                 logging.warning(f"Optimization completed but the last stage did not yield results. Saving results from the last successful stage to {output_filename}. Total time: {total_time:.4f} seconds.")
                 return 1 # Indicate partial success/last stage failure
        else:
             logging.error(f"Failed to save optimization results to {output_filename}. Total time: {total_time:.4f} seconds.")
             return 1 # Indicate failure (save failed)
    else:
        total_time = time.perf_counter() - start_overall_time
        logging.error(f"No valid optimization results were generated to save. Total time: {total_time:.4f} seconds.")
        return 1 # Indicate failure (no results)