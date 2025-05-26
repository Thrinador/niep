import json
import numpy as np
from scipy.stats.qmc import LatinHypercube
import datetime
import multiprocessing
import os

# --- Matrix 1 Definition & Eigenvalue Generation ---
def construct_matrix1(a, b, c):
    return np.array([
        [1 - a - b, a, b, 0, 0],
        [a, 1 - 2 * a - 2 * c, a, c, c],
        [b, a, 1 - a - b, 0, 0],
        [0, c, 0, 0, 1 - c],
        [0, c, 0, 1 - c, 0]
    ])

def get_eigenvalues_lhs_matrix1(num_samples=100000, seed=None):
    print(f"Generating {num_samples} eigenvalue sets for Matrix 1 using LHS...")
    sampler = LatinHypercube(d=3, seed=seed) 
    u_samples = sampler.random(n=num_samples)
    u1, u2, u3 = u_samples[:, 0], u_samples[:, 1], u_samples[:, 2]
    a_params = 0.5 * u1
    c_params = (0.5 - a_params) * u2
    b_params = (1.0 - a_params) * u3
    all_raw_eigenvalues = np.zeros((num_samples, 5))
    progress_report_interval = max(1, num_samples // 10)
    for i in range(num_samples):
        matrix = construct_matrix1(a_params[i], b_params[i], c_params[i])
        all_raw_eigenvalues[i, :] = np.linalg.eigvalsh(matrix)
        if (i + 1) % progress_report_interval == 0:
            print(f"  M1: Generated eigenvalues for {i + 1}/{num_samples} matrices ({((i+1)/num_samples*100):.1f}%)...")
    print("Matrix 1 eigenvalue generation complete.")
    return all_raw_eigenvalues

# --- Matrix 2 Definition & Eigenvalue Generation ---
def construct_matrix2(a, b, c):
    return np.array([
        [1-a-c, 0    , a    , 0    , c    ],
        [0    , 1-a-b, 0    , b    , a    ],
        [a    , 0    , 1-a-b, b    , 0    ],
        [0    , b    , b    , 1-2*b, 0    ],
        [c    , a    , 0    , 0    , 1-a-c]
    ])

def get_eigenvalues_lhs_matrix2(num_samples=100000, seed=None):
    print(f"Generating {num_samples} eigenvalue sets for Matrix 2 using LHS...")
    sampler = LatinHypercube(d=3, seed=seed)
    u_samples = sampler.random(n=num_samples)
    u1, u2, u3 = u_samples[:, 0], u_samples[:, 1], u_samples[:, 2]
    b_params = 0.5 * u1
    a_params = (1.0 - b_params) * u2
    c_params = (1.0 - a_params) * u3
    all_raw_eigenvalues = np.zeros((num_samples, 5))
    progress_report_interval = max(1, num_samples // 10)
    for i in range(num_samples):
        matrix = construct_matrix2(a_params[i], b_params[i], c_params[i])
        all_raw_eigenvalues[i, :] = np.linalg.eigvalsh(matrix)
        if (i + 1) % progress_report_interval == 0:
            print(f"  M2: Generated eigenvalues for {i + 1}/{num_samples} matrices ({((i+1)/num_samples*100):.1f}%)...")
    print("Matrix 2 eigenvalue generation complete.")
    return all_raw_eigenvalues

# --- Matrix 3 Definition & Eigenvalue Generation ---
def construct_matrix3(a, b, c):
    return np.array([
        [1-4*a,   a,       a,       a,       a      ],
        [a,       1-a-b-c, b,       c,       0      ],
        [a,       b,       1-a-2*b, 0,       b      ],
        [a,       c,       0,       1-a-2*c, c      ],
        [a,       0,       b,       c,       1-a-b-c]
    ])

def get_eigenvalues_lhs_matrix3(num_samples=100000, seed=None):
    print(f"Generating {num_samples} eigenvalue sets for Matrix 3 using LHS...")
    sampler = LatinHypercube(d=3, seed=seed)
    u_samples = sampler.random(n=num_samples)
    u1, u2, u3 = u_samples[:, 0], u_samples[:, 1], u_samples[:, 2]
    a_params = 0.25 * u1
    b_max_val = (1.0 - a_params) / 2.0
    b_params = b_max_val * u2
    c_max1 = (1.0 - a_params) / 2.0
    c_max2 = 1.0 - a_params - b_params
    c_max_val = np.minimum(c_max1, c_max2)
    c_max_val[c_max_val < 0] = 0 
    c_params = c_max_val * u3
    all_raw_eigenvalues = np.zeros((num_samples, 5))
    progress_report_interval = max(1, num_samples // 10)
    for i in range(num_samples):
        matrix = construct_matrix3(a_params[i], b_params[i], c_params[i])
        all_raw_eigenvalues[i, :] = np.linalg.eigvalsh(matrix)
        if (i + 1) % progress_report_interval == 0:
            print(f"  M3: Generated eigenvalues for {i + 1}/{num_samples} matrices ({((i+1)/num_samples*100):.1f}%)...")
    print("Matrix 3 eigenvalue generation complete.")
    return all_raw_eigenvalues

# --- Matrix 4 Definition & Eigenvalue Generation ---
def construct_matrix4(a, b, c):
    """Constructs the 5x5 parameterized matrix M4."""
    return np.array([
        [1-2*a-c, a,         a,         c,        0],
        [a,       1-a-b-c,   c,         b,        0],
        [a,       c,         1-a-2*b,   b,        0],
        [c,       b,         b,         1-2*b-c,  0],
        [0,       0,         0,         0,        1]
    ])

def get_eigenvalues_lhs_matrix4(num_samples=100000, seed=None):
    print(f"Generating {num_samples} eigenvalue sets for Matrix 4 using LHS...")
    sampler = LatinHypercube(d=3, seed=seed)
    u_samples = sampler.random(n=num_samples)
    u1, u2, u3 = u_samples[:, 0], u_samples[:, 1], u_samples[:, 2]

    # Parameter transformation for Matrix 4
    # Constraints: a,b,c >= 0; 2a+c <= 1; a+b+c <= 1; a+2b <= 1; 2b+c <= 1
    a_params = 0.5 * u1  # 0 <= a <= 0.5

    b_upper_limit_from_a = (1.0 - a_params) / 2.0  # from a+2b <= 1
    b_upper_limit_overall = 0.5                     # from 2b+c <= 1 (worst case c=0)
    b_params_upper_bound = np.minimum(b_upper_limit_from_a, b_upper_limit_overall)
    b_params = b_params_upper_bound * u2

    c_upper_limit1 = 1.0 - 2*a_params       # from 2a+c <= 1
    c_upper_limit2 = 1.0 - a_params - b_params # from a+b+c <= 1
    c_upper_limit3 = 1.0 - 2*b_params       # from 2b+c <= 1
    c_params_upper_bound = np.minimum(np.minimum(c_upper_limit1, c_upper_limit2), c_upper_limit3)
    c_params_upper_bound[c_params_upper_bound < 0] = 0 # Ensure non-negativity
    c_params = c_params_upper_bound * u3
    
    all_raw_eigenvalues = np.zeros((num_samples, 5))
    progress_report_interval = max(1, num_samples // 10)
    for i in range(num_samples):
        matrix = construct_matrix4(a_params[i], b_params[i], c_params[i])
        all_raw_eigenvalues[i, :] = np.linalg.eigvalsh(matrix)
        if (i + 1) % progress_report_interval == 0:
            print(f"  M4: Generated eigenvalues for {i + 1}/{num_samples} matrices ({((i+1)/num_samples*100):.1f}%)...")
    print("Matrix 4 eigenvalue generation complete.")
    return all_raw_eigenvalues

# --- General Eigenvalue Processing ---
def process_parameterized_eigenvalues(raw_param_eigenvalues, matrix_name="Matrix"):
    # (Unchanged)
    print(f"Processing parameterized eigenvalues for {matrix_name}...")
    num_sets = raw_param_eigenvalues.shape[0]
    processed_eigenvalues_arr = np.zeros((num_sets, 4))
    for i in range(num_sets):
        current_set = raw_param_eigenvalues[i, :]
        idx_to_remove = np.argmin(np.abs(current_set - 1.0))
        remaining_eigenvalues = np.delete(current_set, idx_to_remove)
        remaining_eigenvalues.sort() 
        processed_eigenvalues_arr[i, :] = remaining_eigenvalues
    print(f"Processed {processed_eigenvalues_arr.shape[0]} sets for {matrix_name}, {processed_eigenvalues_arr.shape[1]} eigenvalues each.")
    return processed_eigenvalues_arr

# --- Initial Pre-filtering (Serial) ---
def apply_initial_prefilters(data_list, sum_eigenvalues_tolerance, matrix_entry_threshold=0.05):
    # (Unchanged)
    print(f"\nApplying initial pre-filters...")
    print(f"  Matrix pre-filter: Remove if no matrix entry is < {matrix_entry_threshold}")
    print(f"  Sum-of-eigenvalues pre-filter: Remove if |1 + sum(eigs)| < {sum_eigenvalues_tolerance}")
    kept_after_all_prefilters = []
    count_removed_by_sum_prefilter = 0
    count_removed_by_matrix_prefilter = 0
    count_invalid_item_structure = 0
    for item in data_list:
        if not isinstance(item, dict): count_invalid_item_structure += 1; continue
        has_eigenvalues, has_matrix = "eigenvalues" in item, "matrix" in item
        if not (has_eigenvalues and has_matrix): count_invalid_item_structure += 1; continue
        matrix_entries_raw = item["matrix"]
        if not isinstance(matrix_entries_raw, list): count_invalid_item_structure += 1; continue
        try:
            if not matrix_entries_raw: matrix_entries = np.array([])
            else:
                if not all(isinstance(x, (int, float)) for x in matrix_entries_raw): count_invalid_item_structure +=1; continue
                matrix_entries = np.array(matrix_entries_raw, dtype=float)
        except (ValueError, TypeError): count_invalid_item_structure += 1; continue
        if not matrix_entries_raw or not np.any(matrix_entries < matrix_entry_threshold):
            count_removed_by_matrix_prefilter += 1; continue
        item_eigenvalues_raw = item["eigenvalues"]
        if not isinstance(item_eigenvalues_raw, list) or len(item_eigenvalues_raw) != 4: count_invalid_item_structure += 1; continue
        try: item_eigenvalues = np.array(item_eigenvalues_raw, dtype=float)
        except ValueError: count_invalid_item_structure += 1; continue
        eigenvalue_sum = np.sum(item_eigenvalues)
        if np.abs(1.0 + eigenvalue_sum) < sum_eigenvalues_tolerance:
            count_removed_by_sum_prefilter += 1; continue 
        kept_after_all_prefilters.append(item)
    print(f"Initial pre-filtering complete.")
    print(f"  Items skipped due to invalid structure/type for pre-filters: {count_invalid_item_structure}")
    print(f"  Items removed by matrix pre-filter: {count_removed_by_matrix_prefilter}")
    print(f"  Items removed by sum-of-eigenvalues pre-filter: {count_removed_by_sum_prefilter}")
    return kept_after_all_prefilters, count_removed_by_matrix_prefilter, count_removed_by_sum_prefilter, count_invalid_item_structure

# --- Parallel Filtering Logic ---
# (init_worker_main_filter, process_item_main_filter_parallel, execute_filtering_round_parallel functions remain unchanged)
worker_param_eigenvalues_g = None; worker_epsilon_g = None
def init_worker_main_filter(param_eigenvalues_arg, epsilon_arg):
    global worker_param_eigenvalues_g, worker_epsilon_g
    worker_param_eigenvalues_g = param_eigenvalues_arg; worker_epsilon_g = epsilon_arg
def process_item_main_filter_parallel(item):
    global worker_param_eigenvalues_g, worker_epsilon_g
    try: item_eigenvalues = np.array(item["eigenvalues"], dtype=float)
    except (TypeError, ValueError, KeyError): return item 
    item_eigenvalues.sort()
    diffs = np.abs(worker_param_eigenvalues_g - item_eigenvalues); max_abs_diffs_per_param_set = np.max(diffs, axis=1)
    if np.any(max_abs_diffs_per_param_set < worker_epsilon_g): return None 
    return item 
def execute_filtering_round_parallel(input_data_list, param_eigenvalues_processed, epsilon, round_name="Round"):
    kept_data = []; main_filtered_out_count_this_round = 0; num_items_this_round = len(input_data_list); items_processed_for_progress = 0
    print(f"\nStarting {round_name} main filtering for {num_items_this_round} items..."); print(f"  {round_name} using comparison epsilon: {epsilon}")
    if num_items_this_round == 0: print(f"  No items to filter in {round_name}."); return [], 0
    update_interval = 2000
    if update_interval > num_items_this_round and num_items_this_round > 0: update_interval = max(1, num_items_this_round // 4 if num_items_this_round >=4 else 1)
    num_processes = min(os.cpu_count(), num_items_this_round); num_processes = max(1, num_processes)
    print(f"  {round_name} using {num_processes} worker processes...")
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker_main_filter, initargs=(param_eigenvalues_processed, epsilon)) as pool:
        results_iterator = pool.imap_unordered(process_item_main_filter_parallel, input_data_list)
        for result_item in results_iterator:
            items_processed_for_progress += 1
            if result_item is None: main_filtered_out_count_this_round += 1
            else: kept_data.append(result_item)
            if items_processed_for_progress % update_interval == 0 and items_processed_for_progress < num_items_this_round:
                if update_interval > 1 or num_items_this_round > 20:
                     print(f"  {round_name}: Processed {items_processed_for_progress}/{num_items_this_round} ({((items_processed_for_progress)/num_items_this_round*100):.2f}%). Removed in this round so far: {main_filtered_out_count_this_round}...")
    print(f"{round_name} filtering complete. Items removed by main filter in this round: {main_filtered_out_count_this_round}.")
    return kept_data, main_filtered_out_count_this_round

# --- Main Execution Block ---
if __name__ == '__main__':
    # Configuration
    NUM_PARAM_SAMPLES_M1 = 100000; LHS_SEED_M1 = 42;  EPSILON_THRESHOLD_M1 = 2e-1   
    NUM_PARAM_SAMPLES_M2 = 100000; LHS_SEED_M2 = 123; EPSILON_THRESHOLD_M2 = 2e-1   
    NUM_PARAM_SAMPLES_M3 = 100000; LHS_SEED_M3 = 789; EPSILON_THRESHOLD_M3 = 2e-1 
    NUM_PARAM_SAMPLES_M4 = 100000; LHS_SEED_M4 = 654; EPSILON_THRESHOLD_M4 = 2e-1 # New for M4
    
    PREFILTER_SUM_TOLERANCE = 2e-1 
    MATRIX_ENTRY_PREFILTER_THRESHOLD = 0.02
    
    INPUT_JSON_FILE = 'sniep/data/ds-sniep_n5_dims15_15_15.json' 
    OUTPUT_JSON_FILE = 'filtered_output_data_round4.json' # Updated output file name

    start_time_str = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    print(f"Script started at: {start_time_str}")

    # --- Generate Eigenvalues for All Matrices ---
    raw_eigenvalues_m1 = get_eigenvalues_lhs_matrix1(num_samples=NUM_PARAM_SAMPLES_M1, seed=LHS_SEED_M1)
    processed_eigenvalues_m1 = process_parameterized_eigenvalues(raw_eigenvalues_m1, "Matrix 1")

    raw_eigenvalues_m2 = get_eigenvalues_lhs_matrix2(num_samples=NUM_PARAM_SAMPLES_M2, seed=LHS_SEED_M2)
    processed_eigenvalues_m2 = process_parameterized_eigenvalues(raw_eigenvalues_m2, "Matrix 2")

    raw_eigenvalues_m3 = get_eigenvalues_lhs_matrix3(num_samples=NUM_PARAM_SAMPLES_M3, seed=LHS_SEED_M3)
    processed_eigenvalues_m3 = process_parameterized_eigenvalues(raw_eigenvalues_m3, "Matrix 3")

    raw_eigenvalues_m4 = get_eigenvalues_lhs_matrix4(num_samples=NUM_PARAM_SAMPLES_M4, seed=LHS_SEED_M4) # New
    processed_eigenvalues_m4 = process_parameterized_eigenvalues(raw_eigenvalues_m4, "Matrix 4") # New

    # --- Load Initial JSON Data ---
    try:
        with open(INPUT_JSON_FILE, 'r') as f: initial_json_data = json.load(f)
        if not isinstance(initial_json_data, list): print(f"Error: Input JSON from '{INPUT_JSON_FILE}' is not a list. Exiting."); exit()
        print(f"\nSuccessfully loaded {len(initial_json_data)} items from '{INPUT_JSON_FILE}'.")
    except FileNotFoundError: print(f"Error: Input JSON file not found at '{INPUT_JSON_FILE}'. Exiting."); exit()
    except json.JSONDecodeError: print(f"Error: Could not decode JSON from '{INPUT_JSON_FILE}'. Exiting."); exit()
    initial_item_count = len(initial_json_data)

    # --- Step 1: Initial Pre-filters (Serial) ---
    data_after_initial_prefilters, count_removed_by_matrix_prefilter, \
    count_removed_by_sum_prefilter, count_invalid_items = \
        apply_initial_prefilters(initial_json_data, PREFILTER_SUM_TOLERANCE, matrix_entry_threshold=MATRIX_ENTRY_PREFILTER_THRESHOLD)
    items_entering_round1 = len(data_after_initial_prefilters)

    # --- Step 2: Round 1 Filtering (Parallel, Matrix 1) ---
    data_after_round1, count_removed_round1_main = execute_filtering_round_parallel(
        data_after_initial_prefilters, processed_eigenvalues_m1, EPSILON_THRESHOLD_M1, round_name="Round 1 (Matrix 1)")
    items_entering_round2 = len(data_after_round1)

    # --- Step 3: Round 2 Filtering (Parallel, Matrix 2) ---
    data_after_round2, count_removed_round2_main = execute_filtering_round_parallel(
        data_after_round1, processed_eigenvalues_m2, EPSILON_THRESHOLD_M2, round_name="Round 2 (Matrix 2)")
    items_entering_round3 = len(data_after_round2)

    # --- Step 4: Round 3 Filtering (Parallel, Matrix 3) ---
    data_after_round3, count_removed_round3_main = execute_filtering_round_parallel(
        data_after_round2, processed_eigenvalues_m3, EPSILON_THRESHOLD_M3, round_name="Round 3 (Matrix 3)")
    items_entering_round4 = len(data_after_round3) # New

    # --- Step 5: Round 4 Filtering (Parallel, Matrix 4) ---
    final_kept_data, count_removed_round4_main = execute_filtering_round_parallel( # New
        data_after_round3, processed_eigenvalues_m4, EPSILON_THRESHOLD_M4, round_name="Round 4 (Matrix 4)")
    
    # --- Save Final Data & Print Summary ---
    try:
        with open(OUTPUT_JSON_FILE, 'w') as f: json.dump(final_kept_data, f, indent=4)
        print(f"\nFinal filtered data saved to '{OUTPUT_JSON_FILE}'")
    except IOError: print(f"Error: Could not write final data to output JSON file '{OUTPUT_JSON_FILE}'")

    print("\n--- Filtering Summary ---")
    print(f"Initial items loaded: {initial_item_count}")
    print(f"Items skipped due to invalid structure for pre-filters: {count_invalid_items}")
    print(f"Items removed by matrix pre-filter (no entry < {MATRIX_ENTRY_PREFILTER_THRESHOLD}): {count_removed_by_matrix_prefilter}")
    print(f"Items removed by sum-of-eigenvalues pre-filter (|1 + sum(eigs)| < tol): {count_removed_by_sum_prefilter}")
    print(f"Items entering Round 1 filtering: {items_entering_round1}")
    print(f"Items removed by Round 1 (Matrix 1): {count_removed_round1_main}")
    print(f"Items entering Round 2 filtering: {items_entering_round2}")
    print(f"Items removed by Round 2 (Matrix 2): {count_removed_round2_main}")
    print(f"Items entering Round 3 filtering: {items_entering_round3}")
    print(f"Items removed by Round 3 (Matrix 3): {count_removed_round3_main}")
    print(f"Items entering Round 4 filtering: {items_entering_round4}") # New
    print(f"Items removed by Round 4 (Matrix 4): {count_removed_round4_main}") # New
    print(f"Total items kept: {len(final_kept_data)}")
    
    total_effectively_removed = count_removed_by_matrix_prefilter + count_removed_by_sum_prefilter + \
                                count_removed_round1_main + count_removed_round2_main + \
                                count_removed_round3_main + count_removed_round4_main # New
    print(f"Total items effectively filtered out (pre-filters + all rounds): {total_effectively_removed}")

    end_time_str = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    print(f"\nScript execution finished at: {end_time_str}")