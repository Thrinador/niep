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
    Dynamically loads combined value-and-Jacobian functions.
    """
    try:
        n = config['global_data']['n']
        matrix_type = config['global_data']['matrix_type']
        module_name = f"lib.{matrix_type}_symbolic_minors_n{n}"

        logging.info(f"Loading combined value/jacobian functions from module: {module_name}")
        symbolic_module = importlib.import_module(module_name)

        combined_funcs = []
        # Assuming the generator produces combined value/jac functions
        name_pattern = f"calculate_S{{k}}_n{n}_value_and_jac"

        # Load positive versions (for minimization of S_k)
        for k in range(1, n + 1):
            func_name = name_pattern.format(k=k)
            if hasattr(symbolic_module, func_name):
                combined_funcs.append(getattr(symbolic_module, func_name))
            else:
                logging.error(f"Required function {func_name} not found in {module_name}.py")
                raise ImportError(f"Function {func_name} not found.")

        # Create and append negative versions for maximization (by minimizing -S_k)
        neg_wrapper = lambda f: (lambda x, pf=f: (-pf(x)[0], -pf(x)[1]))
        combined_funcs.extend([neg_wrapper(pf) for pf in combined_funcs[:n]])

        logging.info(f"Successfully loaded {len(combined_funcs)} combined functions.")
        return combined_funcs

    except ImportError as e:
        logging.error(f"Failed to import module or function: {e}")
        return None
    except KeyError as e:
        logging.error(f"Configuration missing expected key 'n' or 'global_data': {e}")
        return None
    except Exception as e:
        logging.exception("An unexpected error occurred during function loading:")
        return None

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
                A[i, idx + i] = 1 # Diagonal elements x_i_i
                for j in range(i + 1, n):
                    A[i, idx + j] = 1 # Off-diagonal elements x_i_j
                    A[j, idx + j] = 1
                idx += n - i

        return LinearConstraint(A, np.zeros(n), np.ones(n))
    except KeyError as e:
        logging.error(f"Config missing 'n' for matrix constraints: {e}")
        return None
    except Exception as e:
        logging.exception("Error building matrix constraints:")
        return None

def run_function_with_const(loc, constraints, combined_func, config, run_count=0):
    """
    Runs scipy.optimize.minimize with 'SLSQP' using a combined function.
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
            logging.error(f"Incorrect matrix types in config: {matrix_type}")
            return None

        tol = config['optimize_data']['tol']
        maxiter = config['optimize_data']['maxiter']
        attempts = config['optimize_data']['attempts']

        func_name = f"S{loc+1}" if loc < n else f"-S{loc-n+1}"
        logging.debug(
            f"Minimize ('SLSQP') for {func_name} (run {run_count}): ftol={tol}, maxiter={maxiter}, attempts={attempts}"
        )

    except KeyError as e:
        logging.error(f"Config missing key for run_function_with_const setup: {e}")
        return None

    bounds = Bounds([0.0] * num_variables, [1.0] * num_variables)

    last_result = None
    for i in range(attempts):
        x0 = np.random.rand(num_variables)
        try:
            result = minimize(
                combined_func, x0, method='SLSQP', jac=True, constraints=constraints,
                bounds=bounds, options={'maxiter': maxiter, 'ftol': tol, 'disp': False,
                                        'eps': 1.49e-08} # <-- ADDED for stability
            )
            last_result = result
            if result.success:
                if run_count % 503 == 0:
                    logging.info(f"Optimization successful for {func_name} run={run_count} on attempt {i+1}.")
                    logging.debug(f"Success result for {func_name} (run {run_count}):\n{result}")
                return result
        except (ValueError, TypeError) as e:
             logging.exception(f"Error during minimize attempt {i+1} for {func_name} run={run_count}: {e}")
             last_result = None; break
        except Exception as e:
             logging.exception(f"Unexpected Error in minimize attempt {i+1} for {func_name} run={run_count}:")
             last_result = None

    if last_result is None or not last_result.success:
        logging.error(f"Optimization failed for {func_name} run={run_count} after {attempts} attempts.")
        logging.debug(f"Failed result for {func_name} (run {run_count}):\n{result}")
        if last_result is None:
            logging.error(f"All attempts failed critically for {func_name} run={run_count}.")

    return last_result

def optimize_func(loc, combined_funcs, config, eqs=[], count=0):
    """Wrapper to set up constraints and call the optimizer."""
    n = config['global_data']['n']
    # --- NEW: Get separate, larger tolerance for nonlinear constraints ---
    # Defaults to a reasonable 1e-6 if not specified in config
    nlc_tol = config.get('optimize_data', {}).get('nlc_tol', 1e-6)

    func_name = f"S{loc+1}" if loc < n else f"-S{loc-n+1}"
    constraints_list = []

    if eqs:
        for eq_func_idx, eq_target_val in eqs:
            if 0 <= eq_func_idx < len(combined_funcs):
                nlc_func = lambda x, idx=eq_func_idx: combined_funcs[idx](x)[0]
                nlc_jac = lambda x, idx=eq_func_idx: combined_funcs[idx](x)[1]
                # --- NEW: Use the larger nlc_tol to create a "band" constraint ---
                constraints_list.append(NonlinearConstraint(
                    nlc_func, lb=eq_target_val - nlc_tol, ub=eq_target_val + nlc_tol, jac=nlc_jac
                ))
            else:
                 logging.error(f"Constraint function index {eq_func_idx} out of bounds.")

    linear_constraints = build_matrix_constraints(config)
    if linear_constraints:
        constraints_list.append(linear_constraints)

    return run_function_with_const(
        loc, constraints_list, combined_funcs[loc], config, count
    )

def _optimize_func_parallel_wrapper(args):
    """Helper for pathos.Pool.imap."""
    loc, combined_funcs, config, eqs, count = args
    return optimize_func(loc, combined_funcs, config, eqs, count)


def build_next_mesh(
    previous_mesh_points: List[Tuple[float, ...]],
    min_results: List[Any],
    max_results: List[Any],
    config: Dict,
    current_dim_index: int
) -> Optional[List[Tuple[float, ...]]]:
    """Builds mesh points for the next dimension, with optional data cleaning."""
    logging.info(f"Building mesh for dimension {current_dim_index + 2}...")
    try:
        points_dim = config['global_data']['points_dim']
        num_new_points = points_dim[current_dim_index]
        tol = config['optimize_data']['tol']
        rounding_precision = config.get('optimize_data', {}).get('optimizer_rounding', None)
        if rounding_precision is not None:
            logging.info(f"Data cleaning enabled: rounding results to {rounding_precision} decimal places.")

    except (KeyError, IndexError) as e:
        logging.error(f"Configuration error accessing points_dim at index {current_dim_index}: {e}")
        return None

    next_mesh_points = []
    valid_source_count = 0

    if len(previous_mesh_points) != len(min_results) or len(previous_mesh_points) != len(max_results):
        logging.error(f"Mismatched input lengths in build_next_mesh.")
        return None

    for i, prev_point_coords in enumerate(previous_mesh_points):
        min_res, max_res = min_results[i], max_results[i]

        is_valid = (min_res and max_res and getattr(min_res, 'success', False) and getattr(max_res, 'success', False))
        if not is_valid:
             logging.warning(f"Invalid/unsuccessful source result at index {i}. Skipping point for next mesh.")
             continue

        min_val = min_res.fun
        max_val = -max_res.fun

        if rounding_precision is not None:
            min_val = round_decimal(min_val, rounding_precision)
            max_val = round_decimal(max_val, rounding_precision)

        if min_val > max_val:
            if np.isclose(min_val, max_val, atol=tol):
                min_val = max_val
            else:
                 logging.warning(f"Cleaned max value ({max_val:.4f}) < min value ({min_val:.4f}). Clamping to min_val.")
                 max_val = min_val
        
        if np.isclose(min_val, max_val, atol=tol):
            new_dim_vals = np.full(num_new_points, min_val)
        elif num_new_points >= 2:
            new_dim_vals = np.linspace(min_val, max_val, num_new_points)
        elif num_new_points == 1:
            new_dim_vals = np.array([(min_val + max_val) / 2.0])
        else:
            new_dim_vals = np.array([])

        if new_dim_vals.size > 0:
             valid_source_count += 1
             for new_val in new_dim_vals:
                 next_mesh_points.append(prev_point_coords + (new_val,))

    if not next_mesh_points:
         logging.warning(f"Mesh generation yielded no points (from {valid_source_count}/{len(previous_mesh_points)} valid sources).")
         return []

    logging.info(f"Built mesh for dimension {current_dim_index + 2} with {len(next_mesh_points)} points.")
    return next_mesh_points


def optimize_constrained_dimension(
    mesh_points: List[Tuple[float, ...]],
    combined_funcs: List, config: Dict,
    constraint_func_indices: List[int], optimize_func_idx: int,
    dimension_label: str
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """Runs min/max optimization for a target function, constrained by values at mesh points."""
    n = config['global_data']['n']
    if mesh_points is None: return None, None
    if not mesh_points: return [], []

    num_points = len(mesh_points)
    min_opt_loc, max_opt_loc = optimize_func_idx, optimize_func_idx + n
    all_args = []
    
    for i, point_coords in enumerate(mesh_points):
        constraints_for_point = list(zip(constraint_func_indices, point_coords))
        all_args.append((min_opt_loc, combined_funcs, config, constraints_for_point, i))
        all_args.append((max_opt_loc, combined_funcs, config, constraints_for_point, i))

    logging.info(f"Starting parallel optimization for {dimension_label}...")
    with Pool() as pool:
        results_from_pool = list(pool.imap(_optimize_func_parallel_wrapper, all_args))
    
    final_min_results = [results_from_pool[i] for i in range(0, len(results_from_pool), 2)]
    final_max_results = [results_from_pool[i] for i in range(1, len(results_from_pool), 2)]
        
    return final_min_results, final_max_results


def process_optimization_result(
    result: Any, result_type: str, constraint_labels_values: Dict[str, float], optimized_label: str
) -> Optional[Dict]:
    """Helper to create the labeled dictionary from a SUCCESSFUL optimization result."""
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

    combined_funcs = load_optimization_functions(config)
    if combined_funcs is None:
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

    # --- Initialization: First dimension ---
    num_x_points = points_dim[0]
    if num_x_points < 1: num_x_points = 1
    
    if num_x_points == 1:
         x_values = np.array([float(n) / 2.0])
    else:
        x_values = np.linspace(0.0, float(n), num_x_points)
    
    current_mesh_points = [(x,) for x in x_values]

    min_results, max_results = None, None
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
            combined_funcs=combined_funcs,
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