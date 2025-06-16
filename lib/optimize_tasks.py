import numpy as np
from scipy.optimize import (
    LinearConstraint,
    NonlinearConstraint,
    minimize,
    Bounds,
)
from math import comb, ceil
import importlib
import logging
import time
from pathos.pools import ProcessPool as Pool
from typing import List, Tuple, Dict, Any, Optional

from lib import file_utils

def round_decimal(value: float, precision: int) -> float:
    """
    Rounds a float to a given decimal precision using standard rounding.
    """
    if precision is None or precision < 0:
        return value
    return round(value, precision)

def load_optimization_functions(config):
    """
    Dynamically loads separate value, jacobian, and hessian functions.
    """
    try:
        n = config['global_data']['n']
        matrix_type = config['global_data']['matrix_type']
        module_name = f"lib.{matrix_type}_symbolic_minors_n{n}"

        logging.info(f"Loading separate Value, Jacobian, & Hessian functions from module: {module_name}")
        symbolic_module = importlib.import_module(module_name)

        funcs = {'val': [], 'jac': [], 'hess': []}
        patterns = {
            'val': f"calculate_S{{k}}_n{n}",
            'jac': f"calculate_S{{k}}_n{n}_jacobian",
            'hess': f"calculate_S{{k}}_n{n}_hessian",
        }

        for k in range(1, n + 1):
            for key, pattern in patterns.items():
                func_name = pattern.format(k=k)
                if hasattr(symbolic_module, func_name):
                    funcs[key].append(getattr(symbolic_module, func_name))
                else:
                    raise ImportError(f"Function {func_name} not found.")

        neg_wrapper = lambda f: (lambda x, pf=f: -pf(x))
        for key in funcs:
            positive_funcs = funcs[key][:n]
            funcs[key].extend([neg_wrapper(pf) for pf in positive_funcs])

        logging.info(f"Successfully loaded {len(funcs['val'])} value, {len(funcs['jac'])} jacobian, and {len(funcs['hess'])} hessian functions.")
        return funcs['val'], funcs['jac'], funcs['hess']

    except Exception as e:
        logging.exception("An unexpected error occurred during function loading:")
        return None, None, None

def build_matrix_constraints(config):
    """Builds the linear constraints for matrix row sums."""
    try:
        n = config['global_data']['n']
        matrix_type = config['global_data']['matrix_type']
        A = None

        if matrix_type == 'niep':
            num_variables = n**2 - n
            A = np.zeros((n, num_variables))
            idx = 0
            for i in range(n):
                for j in range(n-1):
                    A[i, idx] = 1
                    idx += 1
        elif matrix_type == 'sniep':
            num_variables = comb(n, 2)
            A = np.zeros((n, num_variables))
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    A[i, idx] = 1
                    A[j, idx] = 1
                    idx += 1
        elif matrix_type == 'sub_sniep':
            num_variables = comb(n+1, 2)
            A = np.zeros((n, num_variables))
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    A[i, idx] = 1
                    A[j, idx] = 1
                    idx += 1

        return LinearConstraint(A, np.zeros(n), np.ones(n))
    except Exception as e:
        logging.exception("Error building matrix constraints:")
        return None

def run_function_with_const(loc, constraints, val_func, jac_func, hess_func, config, run_count=0):
    """
    Runs a hybrid optimization strategy: tries SLSQP first, then falls back to trust-constr.
    """
    try:
        n = config['global_data']['n']
        matrix_type = config['global_data']['matrix_type']
        num_variables = 0
        if matrix_type == 'niep':
            num_variables = n**2 - n
        elif matrix_type == 'sniep':
            num_variables = comb(n, 2)
        elif matrix_type == 'sub_sniep':
            num_variables = comb(n+1, 2)
        else:
            return None

        tol_slsqp = config['optimize_data']['tol_slsqp']
        tol_trust = config['optimize_data']['tol_trust']
        attempts_trust = config['optimize_data']['attempts_trust']
        attempts_slsqp = config['optimize_data']['attempts_slsqp']
        maxiter = config['optimize_data']['maxiter']
        log_every_n = config['optimize_data']['log_every_n']

        func_name = f"S{loc+1}" if loc < n else f"-S{loc-n+1}"

    except KeyError as e:
        logging.error(f"Config missing key for run_function_with_const setup: {e}")
        return None

    bounds = Bounds([0.0] * num_variables, [1.0] * num_variables)
    last_result = None

    for i in range(attempts_slsqp + attempts_trust):
        x0 = np.random.rand(num_variables)
        method = None
        
        try:
            if i < attempts_slsqp:
                method = 'SLSQP'
                options = {'maxiter': maxiter, 'ftol': tol_slsqp, 'disp': False, 'eps': 1.49e-08}

                result = minimize(
                    val_func, x0, method=method, jac=jac_func, constraints=constraints,
                    bounds=bounds, options=options
                )
            else:
                method = 'trust-constr'
                options = {'maxiter': maxiter, 'gtol': tol_trust, 'disp': False}

                result = minimize(
                    val_func, x0, method=method, jac=jac_func, hess=hess_func,
                    constraints=constraints, bounds=bounds, options=options
                )

            last_result = result
            if result.success:
                if run_count % log_every_n == 0:
                    logging.info(f"Optimization successful for {func_name} with '{method}' (run {run_count}) on attempt {i+1}.")
                return result

        except Exception as e:
             logging.exception(f"Unexpected Error in minimize with '{method}' attempt {i+1} for {func_name}:")
             last_result = None
    
    logging.error(f"Optimization failed for {func_name} after {attempts_slsqp + attempts_trust} total attempts.")
    if last_result:
        logging.debug(f"Final failed result object for {func_name}:\n{last_result}")
        
    return last_result

def optimize_func(loc, val_funcs, jac_funcs, hess_funcs, config, eqs=[], count=0):
    """Wrapper to set up constraints and call the hybrid optimizer."""
    n = config['global_data']['n']
    tol_nlc = config['optimize_data']['tol_nlc']
    constraints_list = []

    if eqs:
        for eq_func_idx, eq_target_val in eqs:
            if 0 <= eq_func_idx < len(val_funcs):
                nlc_func = val_funcs[eq_func_idx]
                nlc_jac = jac_funcs[eq_func_idx]
                constraints_list.append(NonlinearConstraint(
                    nlc_func, lb=eq_target_val - tol_nlc, ub=eq_target_val + tol_nlc, jac=nlc_jac
                ))
            else:
                 logging.error(f"Constraint function index {eq_func_idx} out of bounds.")

    linear_constraints = build_matrix_constraints(config)
    if linear_constraints:
        constraints_list.append(linear_constraints)

    return run_function_with_const(
        loc, constraints_list, val_funcs[loc], jac_funcs[loc], hess_funcs[loc], config, count
    )

def _optimize_func_parallel_wrapper(args):
    """Helper for pathos.Pool.imap."""
    loc, val_funcs, jac_funcs, hess_funcs, config, eqs, count = args
    return optimize_func(loc, val_funcs, jac_funcs, hess_funcs, config, eqs, count)

def build_next_mesh(
    previous_mesh_points: List[Tuple[float, ...]],
    min_results: List[Any],
    max_results: List[Any],
    config: Dict,
    current_dim_index: int
) -> Optional[List[Tuple[float, ...]]]:
    """
    Builds a dynamic mesh for the next dimension.
    The number of points for each new line is scaled based on its length relative
    to the longest line in the set.
    """
    logging.info(f"Building dynamic mesh for dimension {current_dim_index + 2}...")
    try:
        # This is now the *maximum* number of points for the longest line.
        max_points_per_line = config['global_data']['points_dim'][current_dim_index]
        tol = config['optimize_data'].get('slsqp_tol', 1e-8)
        rounding_precision = config['optimize_data']['optimizer_rounding']
    except (KeyError, IndexError) as e:
        logging.error(f"Config error accessing points_dim at index {current_dim_index}: {e}")
        return None

    if len(previous_mesh_points) != len(min_results) or len(previous_mesh_points) != len(max_results):
        logging.error("Mismatched input lengths in build_next_mesh.")
        return None

    # --- Pass 1: Pre-process results to find all min/max ranges and their distances ---
    processed_ranges = []
    for i in range(len(previous_mesh_points)):
        min_res, max_res = min_results[i], max_results[i]

        is_valid = (min_res and max_res and getattr(min_res, 'success', False) and getattr(max_res, 'success', False))
        if not is_valid:
            processed_ranges.append(None)  # Add placeholder to maintain index alignment
            continue

        min_val = round_decimal(min_res.fun, rounding_precision)
        max_val = round_decimal(-max_res.fun, rounding_precision)

        # Clamp values if min > max after rounding, which can happen with small tolerances
        if min_val > max_val:
            if np.isclose(min_val, max_val, atol=tol):
                min_val = max_val
            else:
                logging.warning(f"Cleaned max value ({max_val:.4f}) < min value ({min_val:.4f}). Clamping to min_val.")
                max_val = min_val
        
        processed_ranges.append({'min': min_val, 'max': max_val, 'distance': max_val - min_val})

    # Find the maximum distance among all valid lines to use as a benchmark
    valid_distances = [r['distance'] for r in processed_ranges if r is not None]
    max_distance = max(valid_distances) if valid_distances else 0.0
    logging.info(f"Max distance for scaling new mesh points is {max_distance:.6f}")

    # --- Pass 2: Build the new mesh points with dynamically scaled point counts ---
    next_mesh_points = []
    total_new_points = 0
    for i, prev_point_coords in enumerate(previous_mesh_points):
        range_info = processed_ranges[i]
        
        if range_info is None:
            logging.warning(f"Skipping mesh generation for source point {i} due to invalid/unsuccessful result.")
            continue
        
        min_val, max_val, distance = range_info['min'], range_info['max'], range_info['distance']
        
        num_new_points = 0
        if np.isclose(distance, 0, atol=tol):
            # If the distance is zero, only one point is needed
            num_new_points = 1
        elif max_distance > 0:
            # Scale points based on this line's distance relative to the max distance
            num_new_points = ceil((distance / max_distance) * max_points_per_line)
            # Ensure at least 2 points for any non-zero distance line
            if num_new_points < 2:
                num_new_points = 2
        else: # This covers max_distance == 0, so all distances are effectively 0
            num_new_points = 1
            
        # Generate the coordinate values for the new dimension
        if num_new_points == 1:
            new_dim_vals = np.array([(min_val + max_val) / 2.0])
        else:
            new_dim_vals = np.linspace(min_val, max_val, int(num_new_points))

        total_new_points += len(new_dim_vals)
        for new_val in new_dim_vals:
            next_mesh_points.append(prev_point_coords + (new_val,))

    if not next_mesh_points:
        logging.warning("Mesh generation for the next dimension yielded no points.")
        return []

    logging.info(f"Built mesh for dimension {current_dim_index + 2} with {len(next_mesh_points)} points from {len(valid_distances)} valid sources.")
    return next_mesh_points

def optimize_constrained_dimension(
    mesh_points: List[Tuple[float, ...]],
    val_funcs: List, jac_funcs: List, hess_funcs: List, config: Dict,
    constraint_func_indices: List[int], optimize_func_idx: int,
    dimension_label: str
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """Runs min/max optimization, passing all three function lists."""
    n = config['global_data']['n']
    if mesh_points is None: return None, None
    if not mesh_points: return [], []

    num_points = len(mesh_points)
    min_opt_loc, max_opt_loc = optimize_func_idx, optimize_func_idx + n
    all_args = []
    
    for i, point_coords in enumerate(mesh_points):
        constraints_for_point = list(zip(constraint_func_indices, point_coords))
        all_args.append((min_opt_loc, val_funcs, jac_funcs, hess_funcs, config, constraints_for_point, i))
        all_args.append((max_opt_loc, val_funcs, jac_funcs, hess_funcs, config, constraints_for_point, i))

    logging.info(f"Starting parallel optimization for {dimension_label}...")
    with Pool() as pool:
        results_from_pool = list(pool.imap(_optimize_func_parallel_wrapper, all_args))
    
    final_min_results = [results_from_pool[i] for i in range(0, len(results_from_pool), 2)]
    final_max_results = [results_from_pool[i] for i in range(1, len(results_from_pool), 2)]
        
    return final_min_results, final_max_results

def process_optimization_result(
    result: Any, result_type: str, constraint_labels_values: Dict[str, float], optimized_label: str
) -> Optional[Dict]:
    """Creates a labeled dictionary from a SUCCESSFUL optimization result."""
    if not (result and getattr(result, 'success', False)):
        return None

    coefficients = [constraint_labels_values[label] for label in sorted(constraint_labels_values.keys())]
    
    optimized_value = -result.fun if result_type == "max" else result.fun
    coefficients.append(optimized_value)

    matrix_data = result.x.tolist() if hasattr(result, 'x') else None
    return {"type": result_type, "coefficients": coefficients, "matrix": matrix_data}

def run_optimization(config):
    """Main entry point for the entire optimization process."""
    logging.info("Starting generalized optimization process...")
    start_overall_time = time.perf_counter()

    val_funcs, jac_funcs, hess_funcs = load_optimization_functions(config)
    if val_funcs is None:
        logging.error("Failed to load optimization functions. Aborting.")
        return None

    try:
        funcs_to_optimize_config = config['global_data']['funcs_to_optimize'] 
        points_dim = config['global_data']['points_dim']
        n = config['global_data']['n']
        funcs_sequence = [1] + funcs_to_optimize_config
        s_indices = [f - 1 for f in funcs_sequence]

        if len(points_dim) != len(funcs_to_optimize_config):
             logging.error("Config length mismatch between 'points_dim' and 'funcs_to_optimize'.")
             return None
        logging.info(f"Optimization plan: Full sequence S{funcs_sequence}")
    except KeyError as e:
        logging.error(f"Config missing key in 'global_data': {e}")
        return None

    # --- Initialization ---
    num_x_points = points_dim[0]
    if num_x_points < 1: num_x_points = 1
    x_values = np.linspace(0.0, float(n), num_x_points) if num_x_points > 1 else np.array([float(n) / 2.0])
    current_mesh_points = [(x,) for x in x_values]

    last_successful_stage = 0
    num_stages = len(funcs_to_optimize_config)

    # --- Iterative Optimization Loop ---
    for stage in range(num_stages):
        constraint_indices = s_indices[:stage+1]
        optimize_index = s_indices[stage+1]
        stage_label = f"Stage {stage+1}/{num_stages}"

        logging.info(f"--- Running {stage_label}: Opt S{funcs_sequence[stage+1]} | Con S{funcs_sequence[:stage+1]} ---")
        stage_start_time = time.perf_counter()

        min_results, max_results = optimize_constrained_dimension(
            mesh_points=current_mesh_points,
            val_funcs=val_funcs, jac_funcs=jac_funcs, hess_funcs=hess_funcs,
            config=config,
            constraint_func_indices=constraint_indices,
            optimize_func_idx=optimize_index,
            dimension_label=stage_label
        )
        logging.info(f"--- {stage_label} opt finished in {time.perf_counter() - stage_start_time:.4f}s ---")

        if min_results is None or max_results is None:
            logging.error(f"{stage_label} failed critically. Aborting.")
            break
        
        last_successful_stage = stage + 1

        if stage < num_stages - 1:
            mesh_build_start = time.perf_counter()
            next_mesh_points = build_next_mesh(
                current_mesh_points, min_results, max_results, config,
                current_dim_index = stage
            )
            logging.info(f"--- Mesh build for Stage {stage + 2} finished in {time.perf_counter() - mesh_build_start:.4f}s ---")

            if not next_mesh_points:
                logging.warning(f"Failed to build non-empty mesh for Stage {stage + 2}. Cannot continue.")
                break
            current_mesh_points = next_mesh_points

    # --- Process Final Results ---
    final_results_dicts = []
    if last_successful_stage > 0:
        stage_to_process = last_successful_stage
        logging.info(f"Processing results from last successful stage (Stage {stage_to_process}).")

        final_constraint_labels = [f"S{funcs_sequence[k]}_constraint" for k in range(stage_to_process)]
        final_optimize_label = f"S{funcs_sequence[stage_to_process]}_optimized"

        if len(current_mesh_points) == len(min_results) == len(max_results):
            for i, point_coords in enumerate(current_mesh_points):
                constraints_dict = {label: val for label, val in zip(final_constraint_labels, point_coords)}
                
                min_dict = process_optimization_result(min_results[i], "min", constraints_dict, final_optimize_label)
                if min_dict: final_results_dicts.append(min_dict)

                max_dict = process_optimization_result(max_results[i], "max", constraints_dict, final_optimize_label)
                if max_dict: final_results_dicts.append(max_dict)
        else:
             logging.error("Final mesh points and results lists have mismatched lengths. Cannot process.")

    total_time = time.perf_counter() - start_overall_time
    if not final_results_dicts:
        logging.error(f"Optimization generated no processable results. Last success: Stage {last_successful_stage}/{num_stages}. Time: {total_time:.4f}s.")
        return None
    
    logging.info(f"Optimization finished. Last successful stage: {last_successful_stage}/{num_stages}. Returning {len(final_results_dicts)} results. Total time: {total_time:.4f}s.")
    return final_results_dicts