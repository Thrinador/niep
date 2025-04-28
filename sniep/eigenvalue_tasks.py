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

        real_list = []
        for eig in eigvals_list:
            real_list.append(eig.real)
        eigvals_list = real_list

        eigvals_list.sort(key=lambda x: x, reverse=True) # Sort by real part

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

def run_eigenvalue_computation(config, results_data):
    """
    Computes eigenvalues for the provided optimization results list,
    adds them to the dictionaries, and returns the updated list.
    Returns None on critical failure. Does NOT save results.
    """
    logging.info("--- Starting Eigenvalue Computation ---")
    start_time = time.perf_counter()

    if results_data is None:
        logging.error("Received None for optimization results list. Cannot compute eigenvalues.")
        return None

    combined_eigenvalue_results = []
    processed_count = 0
    skipped_count = 0

    for item in results_data:
        try:
            matrix_list = item.get('matrix')
            result_type = item.get('type', 'unknown')
            coefficients = item.get('coefficients')

            if matrix_list is None:
                 logging.warning("Skipping entry due to missing 'matrix'.")
                 skipped_count += 1
                 continue
            if coefficients is None:
                 logging.warning("Skipping entry due to missing 'coefficients'.")
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

            output_dict = item.copy()
            output_dict["eigenvalues"] = eigvals 

            combined_eigenvalue_results.append(output_dict)
            processed_count += 1

        except Exception as e:
            logging.exception(f"Error processing entry: {item}. Skipping.")
            skipped_count +=1

    logging.info(f"Eigenvalues computed for {processed_count} entries ({skipped_count} skipped).")

    if not combined_eigenvalue_results:
        logging.error("No eigenvalues were successfully computed.")
        return combined_eigenvalue_results

    logging.info(f"--- Eigenvalue Computation Finished successfully in {time.perf_counter() - start_time:.4f} seconds ---")
    return combined_eigenvalue_results