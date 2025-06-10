import numpy as np
import logging
import time
from math import comb

from lib import file_utils

def build_matrix(array, config):
    """Builds the n x n matrix from the 1D array of variables."""
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
        logging.error(f"Invalid matrix type in file eigenvalue_tasks with method build matrix. Got {matrix_type}")
        return None

    if array is None or len(array) != num_variables:
         logging.error(f"Invalid array input for build_matrix. Expected length {num_variables}, got {len(array) if array else 'None'}.")
         return None

    matrix = np.zeros((n, n))
    m = 0

    if matrix_type == 'niep':
        for j in range(n):
            for k in range(0, n):
                if j != k:
                    matrix[j][k] = array[m]
                    m += 1
        row_sums = np.sum(matrix, axis=1)
        for j in range(n):
            matrix[j][j] = 1.0 - row_sums[j] # Use float for diagonal
    elif matrix_type == 'sniep':
        for j in range(n):
            for k in range(j+1, n):
                matrix[j][k] = array[m]
                matrix[k][j] = array[m]
                m += 1
        row_sums = np.sum(matrix, axis=1)
        for j in range(n):
            matrix[j][j] = 1.0 - row_sums[j] # Use float for diagonal
    elif matrix_type == 'sub_sniep':
        for j in range(n):
            for k in range(j, n):
                if j == k:
                    matrix[j][j] = array[m]
                    m += 1
                else:
                    matrix[j][k] = array[m]
                    matrix[k][j] = array[m]
                    m += 1

    if m != num_variables:
            logging.warning(f"Variable counter 'm' ({m}) != expected ({num_variables}) after build.")
    return matrix


def find_eigenvalues(matrix, config):
    """Calculates eigenvalues, sorts descending by real part, removes largest."""
    n = config['global_data']['n']

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

        # Scale the array such that the perron root is 1.
        if perron_root > 0.001:
            eigvals_list = list(map(lambda x: x / perron_root, eigvals_list))

        del eigvals_list[0]
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