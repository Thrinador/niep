import numpy as np
import logging
import time
from math import comb

import file_utils

def build_matrix(array, config):
    """Builds the n x n matrix from the 1D array of variables."""
    try:
        n = config['global_data']['n']
        num_variables = comb(n, 2)
    except KeyError as e:
         logging.error(f"Config missing key for build_matrix: {e}")
         return None

    if array is None or len(array) != num_variables:
         logging.error(f"Invalid array input for build_matrix. Expected length {num_variables}, got {len(array) if array else 'None'}.")
         return None

    matrix = np.zeros((n, n))
    m = 0
    try:
        for j in range(n):
            for k in range(j + 1, n):
                if m < len(array):
                    matrix[j][k] = array[m]
                    matrix[k][j] = array[m]
                    m += 1
                else:
                     logging.error(f"Index 'm' exceeded array bounds during matrix build (symmetric).")
                     return None
        row_sums = np.sum(matrix, axis=1)
        for j in range(n):
            matrix[j][j] = 1.0 - row_sums[j] # Use float for diagonal

        if m != num_variables:
             logging.warning(f"Variable counter 'm' ({m}) != expected ({num_variables}) after build.")

        return matrix

    except Exception as e:
        logging.exception("Unexpected error during matrix construction:")
        return None

def find_eigenvalues(matrix, config):
    """Calculates eigenvalues, sorts descending by real part, removes largest."""
    try:
        n = config['global_data']['n']
    except KeyError as e:
         logging.error(f"Config missing key 'n' for find_eigenvalues: {e}")
         return None

    if not isinstance(matrix, np.ndarray) or matrix.shape != (n, n):
        logging.error(f"Invalid input to find_eigenvalues. Expected {n}x{n} ndarray.")
        return None

    try:
        eigvals = np.linalg.eigvals(matrix)
        eigvals_list = eigvals.tolist()
        eigvals_list.sort(key=lambda x: x.real, reverse=True) # Sort by real part

        if not eigvals_list:
             logging.warning("Eigenvalue calculation resulted in an empty list.")
             return []

        perron_root = eigvals_list[0]
        # Tolerance check for Perron root (adjust tol if needed)
        eig_tol = config.get('global_data', {}).get('eig_tol', 1e-5)
        if not np.isclose(perron_root.real, 1.0, atol=eig_tol) or not np.isclose(perron_root.imag, 0.0, atol=eig_tol):
             logging.warning(f"Largest eigenvalue removed ({perron_root}) is not close to 1 (tol={eig_tol}).")

        del eigvals_list[0] # Remove the largest eigenvalue
        logging.debug(f"Calculated and processed eigenvalues: {eigvals_list}")
        return eigvals_list

    except np.linalg.LinAlgError as e:
        logging.exception("Linear algebra error during eigenvalue calculation:")
        return None
    except Exception as e:
        logging.exception("Unexpected error during eigenvalue calculation:")
        return None

def run_eigenvalue_computation(config):
    """Loads results, computes eigenvalues, saves combined/labeled results."""
    logging.info("--- Starting Eigenvalue Computation ---")
    start_time = time.perf_counter()

    # Load optimization results
    optimization_results_filename = file_utils.build_file_name(config, is_coef=True)
    results_data = file_utils.load_results(optimization_results_filename)
    if results_data is None:
        logging.error("Failed to load optimization results. Cannot compute eigenvalues.")
        return 1 # Failure

    # Compute Eigenvalues
    combined_eigenvalue_results = []
    processed_count = 0
    skipped_count = 0

    for item in results_data:
        try:
            matrix_list = item.get('matrix')
            result_type = item.get('type', 'unknown')

            if matrix_list is None:
                 logging.warning("Skipping entry due to missing 'matrix'.")
                 skipped_count += 1
                 continue

            matrix = build_matrix(matrix_list, config)
            if matrix is None:
                logging.warning(f"Skipping {result_type} entry: matrix build failed.")
                skipped_count += 1
                continue

            eigvals = find_eigenvalues(matrix, config)
            if eigvals is None:
                logging.warning(f"Skipping {result_type} entry: eigenvalue calculation failed.")
                skipped_count += 1
                continue

            eigvals_serializable = [e for e in eigvals]

            # Create labeled dictionary, include original constraints/optimized value
            eigenvalue_dict = {
                "type": result_type,
                 # Copy constraint/optimized keys from the optimization result item
                **{k: v for k, v in item.items() if k.endswith(('_constraint', '_optimized'))},
                "eigenvalues": eigvals_serializable,
                "matrix": matrix_list # Include original matrix list
            }
            combined_eigenvalue_results.append(eigenvalue_dict)
            processed_count += 1

        except Exception as e:
            logging.exception(f"Error processing entry: {item}. Skipping.")
            skipped_count +=1

    logging.info(f"Eigenvalues computed for {processed_count} entries ({skipped_count} skipped).")

    if not combined_eigenvalue_results:
        logging.error("No eigenvalues were successfully computed.")
        return 1 # Failure

    # Save Combined Eigenvalues
    eigenvalue_output_filename = file_utils.build_file_name(config, is_coef=False)
    save_success = file_utils.save_results(combined_eigenvalue_results, eigenvalue_output_filename)

    if save_success:
        logging.info(f"--- Eigenvalue Computation Finished successfully in {time.perf_counter() - start_time:.4f} seconds ---")
        return 0 # Success
    else:
        logging.error("Failed to save eigenvalue results.")
        return 1 # Failure