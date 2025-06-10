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

def load_optimization_functions(config):
    """Dynamically loads minors and jacobians based on config"""
    try:
        n = config['global_data']['n']
        matrix_type = config['global_data']['matrix_type']
        module_name = f"{matrix_type}_symbolic_minors_n{n}"

        logging.info(f"Attempting to load functions from module: {module_name}")
        symbolic_module = getattr(importlib.import_module('optimizer_helper_code'), module_name)

        funcs = {'minors': [], 'jacobians': []}
        func_types = {
            'minors': (f"calculate_S{{k}}_n{n}", lambda f: (lambda x, pf=f: -pf(x))),
            'jacobians': (f"calculate_S{{k}}_n{n}_jacobian", lambda f: (lambda x, pf=f: -np.array(pf(x)))),
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

            negative_funcs = [neg_wrapper(pf) for pf in positive_funcs if pf is not None] if positive_funcs else []
            funcs[type_key] = positive_funcs + negative_funcs
            logging.info(f"Successfully loaded {len(positive_funcs)} positive and {len(negative_funcs)} negative {type_key} functions.")

        num_expected = 2 * n
        for type_key in funcs:
            if len(funcs[type_key]) != num_expected:
                logging.warning(f"Loaded {len(funcs[type_key])} {type_key} functions, expected {num_expected}. Check symbolic module.")
                while len(funcs[type_key]) < num_expected:
                    funcs[type_key].append(None)

        return funcs['minors'], funcs['jacobians']

    except ImportError as e:
        logging.error(f"Failed to import module or function: {e}")
        return None, None
    except KeyError as e:
        logging.error(f"Configuration missing expected key 'n' or 'global_data': {e}")
        return None, None
    except Exception as e:
        logging.exception("An unexpected error occurred during function loading:")
        return None, None

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
    except KeyError as e:
        logging.error(f"Config missing 'n' for matrix constraints: {e}")
        return None
    except Exception as e:
        logging.exception("Error building matrix constraints:")
        return None

def run_function_with_const(loc, constraints, funcs_minors, funcs_jacobians, config, run_count=0):
    """
    Runs scipy.optimize.minimize with 'SLSQP', attempting multiple
    random starting points if the first fails.
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
            logging.error(f"Incorrect matrix types in file optimize_tasks function run_function_with_const with matrix type {matrix_type}")
            return None

        tol = config['optimize_data']['tol']
        maxiter = config['optimize_data']['maxiter']
        attempts = config['optimize_data']['attempts']

        func_name = f"S{loc+1}" if loc < n else f"-S{loc-n+1}"
        logging.debug(
            f"Minimize ('SLSQP') for {func_name} (run {run_count}): ftol={tol}, maxiter={maxiter}, attempts={attempts}"
        )

    except KeyError as e:
        logging.error(f"Config missing key for run_function_with_const setup (n): {e}")
        return None
    except Exception as e:
        logging.exception(f"Error setting up minimize parameters:")
        return None

    obj_func = funcs_minors[loc]
    bounds = Bounds([0.0] * num_variables, [1.0] * num_variables)
    jac = funcs_jacobians[loc]

    last_result = None
    for i in range(attempts):
        x0 = np.random.rand(num_variables)
        try:
            result = minimize(
                obj_func, x0, method='SLSQP', jac=jac, constraints=constraints,
                bounds=bounds, options={'maxiter': maxiter, 'ftol': tol, 'disp': False}
            )
            last_result = result
            if result.success:
                if run_count % 503 == 0:
                    logging.info(f"Optimization successful for {func_name} run={run_count} on attempt {i+1}.")
                logging.debug(f"Success Result for {func_name} (run {run_count}):\n{result}")
                return result
        except (ValueError, TypeError) as e:
             logging.exception(f"Error during minimize attempt {i+1} for {func_name} run={run_count}: {e}. Check function/jacobian.")
             last_result = None; break
        except Exception as e:
             logging.exception(f"Unexpected Error in minimize attempt {i+1} for {func_name} run={run_count}:")
             last_result = None

    if last_result is None or not last_result.success:
        logging.error(f"Optimization failed for {func_name} run={run_count} after {attempts} attempts.")
        if last_result is None:
            logging.error(f"All attempts failed critically for {func_name} run={run_count}.")

    return last_result

def optimize_func(loc, funcs_minors, funcs_jacobians, config, eqs=[], count=0):
    """Wrapper to set up constraints and call the optimizer."""
    n = config['global_data']['n']
    tol = config['optimize_data']['tol']

    func_name = f"S{loc+1}" if loc < n else f"-S{loc-n+1}"
    constraints_list = []

    # Nonlinear constraints
    if eqs:
        for eq_func_idx, eq_target_val in eqs:
            if 0 <= eq_func_idx < len(funcs_minors):
                nlc_func = lambda x, idx=eq_func_idx: funcs_minors[idx](x)
                nlc_jac = funcs_jacobians[eq_func_idx] if funcs_jacobians and 0 <= eq_func_idx < len(funcs_jacobians) else '2-point'
                constraints_list.append(NonlinearConstraint(
                    nlc_func, lb=eq_target_val - tol, ub=eq_target_val + tol, jac=nlc_jac
                ))
            else:
                 logging.error(f"Constraint function index {eq_func_idx} out of bounds for {func_name} run {count}. Skipping.")

    # Linear constraints
    linear_constraints = build_matrix_constraints(config)
    if linear_constraints:
        constraints_list.append(linear_constraints)

    # Call the core optimizer
    return run_function_with_const(
        loc, constraints_list, funcs_minors, funcs_jacobians, config, count
    )

def _optimize_func_parallel_wrapper(args):
    """Helper for pathos.Pool.imap"""
    loc, funcs_minors, funcs_jacobians, config, eqs, count = args
    return optimize_func(loc, funcs_minors, funcs_jacobians, config, eqs, count)


def build_next_mesh(
    previous_mesh_points: List[Tuple[float, ...]],
    min_results: List[Any],
    max_results: List[Any],
    config: Dict,
    current_dim_index: int # Index (0-based) in points_dim for the dimension being added
) -> Optional[List[Tuple[float, ...]]]:
    """Builds mesh points for the next dimension."""
    logging.info(f"Building mesh for dimension {current_dim_index + 2} (based on points_dim index {current_dim_index})...")
    try:
        points_dim = config['global_data']['points_dim']
        if current_dim_index >= len(points_dim):
            logging.error(f"'points_dim' ({points_dim}) too short for mesh stage index {current_dim_index}.")
            return None
        num_new_points = points_dim[current_dim_index]
        tol = config['optimize_data']['tol']
    except KeyError as e:
        logging.error(f"Config missing key for build_next_mesh (points_dim): {e}")
        return None
    except IndexError:
        logging.error(f"points_dim config access error at index {current_dim_index}.")
        return None

    next_mesh_points = []
    valid_source_count = 0

    if len(previous_mesh_points) != len(min_results) or len(previous_mesh_points) != len(max_results):
        logging.error(f"Mismatched input lengths in build_next_mesh ({len(previous_mesh_points)}, {len(min_results)}, {len(max_results)})")
        return None

    logging.debug(f"Generating {num_new_points} points in new dimension for each of the {len(previous_mesh_points)} previous points.")

    for i, prev_point_coords in enumerate(previous_mesh_points):
        min_res, max_res = min_results[i], max_results[i]

        is_valid = (min_res is not None and max_res is not None and
                    hasattr(min_res, 'fun') and hasattr(max_res, 'fun') and
                    min_res.fun is not None and max_res.fun is not None and
                    getattr(min_res, 'success', False) and getattr(max_res, 'success', False))

        if not is_valid:
             coord_str = ", ".join(f"{c:.4f}" for c in prev_point_coords)
             logging.warning(f"Invalid/unsuccessful source result at index {i} for coords ({coord_str}). Skipping point for next mesh.")
             continue

        min_val, max_val = min_res.fun, max_res.fun * -1

        if min_val > max_val:
            if np.isclose(min_val, max_val, atol=tol): min_val = max_val
            else:
                 coord_str = ", ".join(f"{c:.4f}" for c in prev_point_coords)
                 logging.warning(f"Max value ({max_val:.4f}) < Min value ({min_val:.4f}) for coords ({coord_str}). Using bounds anyway.")

        if np.isclose(min_val, max_val, atol=tol):
            new_dim_vals = np.full(num_new_points, min_val)
        elif num_new_points >= 2:
            new_dim_vals = np.linspace(min_val, max_val, num_new_points)
        elif num_new_points == 1:
            new_dim_vals = np.array([(min_val + max_val) / 2.0])
        else: new_dim_vals = np.array([])

        if new_dim_vals.size > 0:
             valid_source_count += 1
             for new_val in new_dim_vals:
                 next_mesh_points.append(prev_point_coords + (new_val,))

    if not next_mesh_points:
         logging.warning(f"Mesh generation for dimension {current_dim_index + 2} yielded no points (from {valid_source_count}/{len(previous_mesh_points)} valid sources).")
         return [] if previous_mesh_points else None

    logging.info(f"Built mesh for dimension {current_dim_index + 2} with {len(next_mesh_points)} points (from {valid_source_count} valid sources).")
    return next_mesh_points


def optimize_constrained_dimension(
    mesh_points: List[Tuple[float, ...]],
    funcs_minors: List, funcs_jacobians: List, config: Dict,
    constraint_func_indices: List[int], optimize_func_idx: int,
    dimension_label: str
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """Runs min/max optimization for a target function, constrained by values at mesh points."""
    n = config['global_data']['n']
    if mesh_points is None:
        logging.error(f"Invalid mesh_points input (None) provided ({dimension_label}).")
        return None, None
    if not mesh_points:
        logging.warning(f"Mesh points list for {dimension_label} is empty. No tasks.")
        return [], []

    num_points = len(mesh_points)
    min_opt_loc, max_opt_loc = optimize_func_idx, optimize_func_idx + n
    optimize_func_num = optimize_func_idx + 1

    all_args = []
    skipped_indices = set()
    run_counter = 0
    for i, point_coords in enumerate(mesh_points):
        if len(point_coords) != len(constraint_func_indices):
            logging.error(f"Coord/Constraint mismatch at index {i} ({len(point_coords)} vs {len(constraint_func_indices)}) for {dimension_label}. Skipping point {point_coords}.")
            skipped_indices.add(i); continue

        constraints_for_point = [[idx, val] for idx, val in zip(constraint_func_indices, point_coords)]
        all_args.append((min_opt_loc, funcs_minors, funcs_jacobians, config, constraints_for_point, run_counter, i))
        run_counter += 1
        all_args.append((max_opt_loc, funcs_minors, funcs_jacobians, config, constraints_for_point, run_counter, i))
        run_counter += 1

    if not all_args:
        logging.warning(f"No valid optimization tasks generated for {dimension_label} (all points skipped?).")
        return [None] * num_points, [None] * num_points

    num_tasks = len(all_args)
    constraint_nums_str = ", ".join(map(str, [idx + 1 for idx in constraint_func_indices]))
    logging.info(f"Starting parallel optimization for {dimension_label} "
                 f"(Optimize S{optimize_func_num} | Constraints S[{constraint_nums_str}]).")
    logging.info(f"Running {num_tasks} tasks ({num_tasks // 2} points).")

    start_time = time.perf_counter()
    results_from_pool = []
    try:
        args_for_pool = [arg[:-1] for arg in all_args] # Remove original index 'i' for wrapper
        with Pool() as pool:
            results_from_pool = list(pool.imap(_optimize_func_parallel_wrapper, args_for_pool, chunksize=1))
        end_time = time.perf_counter()
        logging.info(f"Finished {dimension_label} parallel execution in {end_time - start_time:.4f}s.")
    except Exception as e:
        logging.exception(f"Error during parallel processing in {dimension_label}:")
        results_from_pool = [None] * len(all_args)

    # Reconstruct results
    final_min_results = [None] * num_points
    final_max_results = [None] * num_points
    processed_pairs = 0
    successful_pairs = 0

    if len(results_from_pool) != len(all_args):
        logging.error(f"Pool result count ({len(results_from_pool)}) != expected ({len(all_args)}) for {dimension_label}.")
        return [None] * num_points, [None] * num_points

    for idx in range(0, len(all_args), 2):
        original_index = all_args[idx][-1]
        min_res, max_res = results_from_pool[idx], results_from_pool[idx + 1]
        final_min_results[original_index] = min_res
        final_max_results[original_index] = max_res
        processed_pairs += 1
        if min_res and getattr(min_res, 'success', False) and max_res and getattr(max_res, 'success', False):
             successful_pairs += 1

    logging.info(f"Reconstructed results for {processed_pairs} pairs ({dimension_label}). "
                 f"{successful_pairs} successful. {len(skipped_indices)} initially skipped.")

    if len(final_min_results) != num_points or len(final_max_results) != num_points:
         logging.error(f"Result list length mismatch after reconstruction ({dimension_label}).")
         return None, None

    return final_min_results, final_max_results


def process_optimization_result(
    result: Any, result_type: str, constraint_labels_values: Dict[str, float], optimized_label: str
) -> Optional[Dict]:
    """Helper to create the labeled dictionary from a SUCCESSFUL optimization result."""
    if result is None or not getattr(result, 'success', False) or not hasattr(result, 'fun') or result.fun is None:
        if result is not None and not getattr(result, 'success', False):
             logging.debug(f"Skipping processing unsuccessful result ({result_type} {optimized_label}). Msg: {getattr(result, 'message', 'N/A')}")
        elif result is not None:
             logging.warning(f"Invalid successful result object for {result_type} {optimized_label}: {result}")
        return None

    coefficients = []
    sorted_constraint_labels = sorted(constraint_labels_values.keys())
    for label in sorted_constraint_labels:
         coefficients.append(constraint_labels_values[label])

    optimized_value = result.fun * (-1 if result_type == "max" else 1)
    coefficients.append(optimized_value)

    matrix_data = result.x.tolist() if hasattr(result, 'x') and result.x is not None else None
    if matrix_data is None and getattr(result, 'success', False): # Only warn if successful but no matrix
         logging.warning(f"Successful result missing 'x' attribute for {result_type} {optimized_label}.")

    return {"type": result_type, "coefficients": coefficients, "matrix": matrix_data}


def run_optimization(config):
    logging.info("Starting generalized optimization process...")
    start_overall_time = time.perf_counter()

    funcs_minors, funcs_jacobians = load_optimization_functions(config)
    if funcs_minors is None or funcs_jacobians is None:
        logging.error("Failed to load optimization functions. Aborting.")
        return None

    try:
        funcs_to_optimize_config = config['global_data']['funcs_to_optimize'] 
        points_dim = config['global_data']['points_dim']
        n = config['global_data']['n']

        funcs_sequence = [1] + funcs_to_optimize_config
        s_indices = [f - 1 for f in funcs_sequence]

        # Validate lengths based on config lists
        if not isinstance(funcs_to_optimize_config, list):
             logging.error("'funcs_to_optimize' must be a list in config.")
             return None
        if not isinstance(points_dim, list):
             logging.error("'points_dim' must be a list in config.")
             return None
        # Number of stages/dimensions to optimize *after* the first one
        if len(points_dim) != len(funcs_to_optimize_config):
             logging.error(f"Config length mismatch: len(points_dim)={len(points_dim)} != len(funcs_to_optimize)={len(funcs_to_optimize_config)}.")
             return None

        logging.info(f"Optimization plan: Full sequence S{funcs_sequence}, Indices {s_indices}")
        logging.info(f"Mesh points per optimized dimension: {points_dim}")

    except KeyError as e:
        logging.error(f"Config missing key in 'global_data' (funcs_to_optimize, points_dim, n): {e}")
        return None
    except Exception as e:
         logging.exception("Error processing configuration:")
         return None

    # --- Initialization: First dimension (constraint S1) ---
    num_x_points = points_dim[0] # Points along the *second* dimension's axis (constrained by S1)
    # Label for the first constraint axis (S1)
    label_constraint1 = f"S{funcs_sequence[0]}_constraint" # Should always be S1_constraint

    if num_x_points < 1: num_x_points = 1; logging.warning("Points for first optimized dim < 1, set to 1.")

    if num_x_points == 1:
         single_x_value = float(n) / 2.0
         x_values = np.array([single_x_value])
         logging.info(f"Running for single initial constraint value: {label_constraint1}={x_values[0]:.4f}")
    else:
        x_min = 0.0
        x_max = float(n)
        x_values = np.linspace(x_min, x_max, num_x_points)
        logging.info(f"Running for {num_x_points} initial values of {label_constraint1} [{x_min:.4f}, {x_max:.4f}]")

    # Mesh points defining the constraints for the *first* optimization stage (optimizing S2)
    current_mesh_points = [(x,) for x in x_values] # Each tuple represents (S1_value,)

    min_results, max_results = None, None
    last_successful_stage = 0
    # Number of optimization stages is the length of funcs_to_optimize_config
    num_stages = len(funcs_to_optimize_config)

    # --- Iterative Optimization Loop ---
    for stage in range(num_stages):
        # Overall dimension index (1=S1, 2=S2, ...)
        # Constraint functions are up to and including the current stage index in s_indices
        constraint_indices = s_indices[:stage+1]        # Indices: [0], then [0, 1], then [0, 1, 2]
        # Optimize function is the next one in the sequence
        optimize_index = s_indices[stage+1]             # Indices: 1, then 2, then 3
        # Stage label (1-based for user clarity)
        stage_label = f"Stage {stage+1}/{num_stages}"   # Stage 1, Stage 2, Stage 3

        # Labels for results processing
        constraint_labels = [f"S{funcs_sequence[k]}_constraint" for k in range(stage+1)]
        optimize_label = f"S{funcs_sequence[stage+1]}_optimized"
        # Logging info
        constraint_nums_str = ", ".join(map(str, funcs_sequence[:stage+1]))
        optimize_num_str = funcs_sequence[stage+1]

        logging.info(f"--- Running {stage_label}: Opt S{optimize_num_str} | Con S[{constraint_nums_str}] ---")
        stage_start_time = time.perf_counter()

        min_results, max_results = optimize_constrained_dimension(
            current_mesh_points, funcs_minors, funcs_jacobians, config,
            constraint_indices, optimize_index, stage_label
        )
        logging.info(f"--- {stage_label} opt finished in {time.perf_counter() - stage_start_time:.4f}s ---")

        if min_results is None or max_results is None or \
           len(min_results) != len(current_mesh_points) or len(max_results) != len(current_mesh_points):
            logging.error(f"{stage_label} failed critically or returned mismatched results. Aborting.")
            break

        last_successful_stage = stage + 1 # Mark stage N as successfully run

        # --- Prepare for next stage (if any) ---
        if stage < num_stages - 1:
            mesh_build_start = time.perf_counter()
            # Build mesh using points_dim[stage+1] points for the next dimension
            next_mesh_points = build_next_mesh(
                current_mesh_points, min_results, max_results, config,
                current_dim_index = stage + 1 # Index in points_dim
            )
            logging.info(f"--- Mesh build for Stage {stage + 2} finished in {time.perf_counter() - mesh_build_start:.4f}s ---")

            if next_mesh_points is None:
                logging.error(f"Failed critically building mesh for Stage {stage + 2}. Aborting.")
                break
            if not next_mesh_points:
                 logging.warning(f"Mesh for Stage {stage + 2} is empty. Cannot proceed.")
                 break
            current_mesh_points = next_mesh_points

    # --- Process Final Results ---
    final_results_dicts = []
    if last_successful_stage > 0:
        # Process results from 'last_successful_stage' (1-based)
        logging.info(f"Processing results from last successful stage (Stage {last_successful_stage}).")

        # Labels correspond to the constraints *used in* the last successful stage
        final_constraint_indices = s_indices[:last_successful_stage]
        final_optimize_index = s_indices[last_successful_stage]
        final_constraint_labels = [f"S{funcs_sequence[k]}_constraint" for k in range(last_successful_stage)]
        final_optimize_label = f"S{funcs_sequence[last_successful_stage]}_optimized"

        if len(current_mesh_points) != len(min_results) or len(current_mesh_points) != len(max_results):
            logging.error(f"FATAL: Mismatch between final mesh points ({len(current_mesh_points)}) and results ({len(min_results)}, {len(max_results)}) before processing.")
        else:
            num_final_points = len(current_mesh_points)
            processed_count, skipped_count = 0, 0
            logging.debug(f"Processing {num_final_points} points from Stage {last_successful_stage}.")
            for i in range(num_final_points):
                point_coords = current_mesh_points[i]
                if len(point_coords) != len(final_constraint_labels):
                     logging.error(f"Coord length {len(point_coords)} != Label count {len(final_constraint_labels)} at final processing index {i}. Skipping.")
                     if min_results[i] is not None: skipped_count += 1
                     if max_results[i] is not None: skipped_count += 1
                     continue

                # Create {label: value} dict ensuring consistent coefficient order via sorted labels
                sorted_final_labels = sorted(final_constraint_labels)
                constraints_dict = {
                    label: point_coords[final_constraint_labels.index(label)]
                    for label in sorted_final_labels
                }

                min_dict = process_optimization_result(min_results[i], "min", constraints_dict, final_optimize_label)
                if min_dict: processed_count += 1; final_results_dicts.append(min_dict)
                elif min_results[i] is not None: skipped_count += 1

                max_dict = process_optimization_result(max_results[i], "max", constraints_dict, final_optimize_label)
                if max_dict: processed_count += 1; final_results_dicts.append(max_dict)
                elif max_results[i] is not None: skipped_count += 1

            logging.info(f"Processed {processed_count} final result dictionaries ({skipped_count} individual results skipped/failed).")

    # --- Final Logging and Return ---
    total_time = time.perf_counter() - start_overall_time
    if not final_results_dicts:
        logging.error(f"Optimization generated no processable results. Last success: Stage {last_successful_stage}/{num_stages}. Time: {total_time:.4f}s.")
        return None
    elif last_successful_stage < num_stages:
         logging.warning(f"Optimization ended prematurely after Stage {last_successful_stage}/{num_stages}. Returning {len(final_results_dicts)} results. Time: {total_time:.4f}s.")
    else:
         logging.info(f"Optimization completed {num_stages} stages. Returning {len(final_results_dicts)} results. Time: {total_time:.4f}s.")
    return final_results_dicts