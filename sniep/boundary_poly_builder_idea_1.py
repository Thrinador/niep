import json
import numpy as np
from scipy.spatial import KDTree
from sklearn.preprocessing import PolynomialFeatures
import random 
import time 
from math import comb 
import os 

# Assume load_eigenvalue_data, find_implicit_polynomial, and 
# format_polynomial functions are defined as previously.
# They remain unchanged. Re-adding them here for completeness.

def load_eigenvalue_data(json_filepath):
    """Loads 4D eigenvalue data from a JSON file."""
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        eigenvalue_points = []
        for item in data:
            if 'eigenvalues' in item and len(item['eigenvalues']) == 4:
                eigenvalue_points.append(item['eigenvalues'])
            else:
                # print(f"Warning: Skipping item due to missing/incorrect 'eigenvalues': {item}") # Reduce noise
                pass
        if not eigenvalue_points:
            print("Error: No valid eigenvalue data found.")
            return None
        return np.array(eigenvalue_points, dtype=float) 
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return None

def find_implicit_polynomial(points, degree):
    """Finds coefficients of an implicit polynomial P(x)=0 fit."""
    if points is None or points.shape[0] < 1 or points.shape[1] != 4:
        return None, None
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    try:
        A = poly.fit_transform(points)
        num_points, num_coeffs = A.shape
        if num_points < num_coeffs:
             pass             
        _, _, Vh = np.linalg.svd(A)
        coefficients = Vh[-1, :]
        coefficients /= np.linalg.norm(coefficients) # Normalize
        return coefficients, poly
    except np.linalg.LinAlgError:
        return None, None
    except Exception:
        return None, None

def format_polynomial(coefficients, poly_transformer, variables=['x1', 'x2', 'x3', 'x4'], precision=3, coeff_threshold=1e-6):
    """Formats a polynomial equation string from coefficients and features."""
    if coefficients is None: return "Invalid Polynomial (coefficients are None)" 
    feature_names = poly_transformer.get_feature_names_out(variables)
    terms = []
    for coeff, name in zip(coefficients, feature_names):
        abs_coeff = abs(coeff)
        if abs_coeff < coeff_threshold: continue
        sign = "-" if coeff < 0 else "+"
        if np.isclose(abs_coeff, 1.0, atol=coeff_threshold):
            coeff_str = ""
            if name == '1': coeff_str = "1.0" 
        else:
            coeff_str = f"{abs_coeff:.{precision}f}"
        if name == '1':
            term_str = f"{coeff_str}"
        else:
             mult_sign = "*" if coeff_str else "" 
             term_str = f"{coeff_str}{mult_sign}{name}"
        terms.append((sign, term_str))

    if not terms: return "0 = 0 (All coefficients below threshold)"
    first_sign, first_term = terms[0]
    poly_str = f"{'-' if first_sign == '-' else ''}{first_term}"
    for sign, term in terms[1:]:
        poly_str += f" {sign} {term}"
    return f"{poly_str} = 0"


# --- Updated Adaptive Fitting Function with Progress Check ---

def adaptive_polynomial_boundary_estimation(json_filepath, n_neighbors, poly_degree, tolerance, 
                                            prefilter_poly=True, random_seed=None,
                                            min_points_covered_threshold=1,
                                            min_side_percentage=0.75, 
                                            sign_check_tolerance=1e-6,
                                            max_iterations_without_progress=None): # New parameter
    """
    Fits polynomials, checks sign consistency, filters results, and monitors progress.

    Args:
        json_filepath (str): Path to JSON file.
        n_neighbors (int): Number of neighbors for local fit.
        poly_degree (int): Degree of polynomial fit.
        tolerance (float): Tolerance |P(x)| for global coverage check.
        prefilter_poly (bool): Apply pre-filter based on 1+x1+..+x4=0.
        random_seed (int, optional): Seed for RNG.
        min_points_covered_threshold (int): Min points covered globally to keep polynomial.
        min_side_percentage (float): Min percentage of LOCAL points required on one side.
        sign_check_tolerance (float): Tolerance for sign check.
        max_iterations_without_progress (int, optional): Max iterations allowed without
                                                         covering any new points. 
                                                         Defaults to 2 * num_total_points.

    Returns:
        list: Filtered list of polynomial fit dictionaries.
    """
    # --- Seed Random Generator ---
    if random_seed is not None:
        print(f"Using random seed: {random_seed}")
        random.seed(random_seed)

    # 1. Load Data
    all_points = load_eigenvalue_data(json_filepath)
    if all_points is None or all_points.shape[0] == 0: return []
    num_total_points = all_points.shape[0]
    print(f"Loaded {num_total_points} points.")
    
    # --- Set Default for max_iterations_without_progress ---
    if max_iterations_without_progress is None:
        max_iterations_without_progress = num_total_points * 2 # Default heuristic
        print(f"Setting max_iterations_without_progress to {max_iterations_without_progress}")
    elif max_iterations_without_progress <= 0:
         print(f"Warning: max_iterations_without_progress ({max_iterations_without_progress}) must be positive. Using default.")
         max_iterations_without_progress = num_total_points * 2
         
    # --- Parameter Validation (other params) ---
    # (Previous validations...)
    if not (0.5 <= min_side_percentage <= 1.0):
        min_side_percentage = max(0.5, min(1.0, min_side_percentage))
    if not (sign_check_tolerance >= 0):
         sign_check_tolerance = abs(sign_check_tolerance)
    if not (min_points_covered_threshold >= 1):
        min_points_covered_threshold = 1
    #... other checks

    # --- Initialization ---
    uncovered_indices = set(range(num_total_points))
    initial_polynomial_fits = [] 
    iterations_since_last_progress = 0 # New counter
    
    # --- Optional Pre-filtering Step ---
    pre_covered_indices = set() 
    if prefilter_poly:
        # (Pre-filtering logic as before...)
        print("Applying pre-filter...")
        try:
             p0_values = 1.0 + np.sum(all_points[:, :4], axis=1)
             abs_p0_values = np.abs(p0_values)
             pre_covered_mask = abs_p0_values <= tolerance
             pre_covered_indices = set(np.where(pre_covered_mask)[0]) 
             num_pre_covered = len(pre_covered_indices)
             if num_pre_covered > 0:
                 print(f"Pre-filter: Found {num_pre_covered} points.")
                 uncovered_indices.difference_update(pre_covered_indices)
                 print(f"            {len(uncovered_indices)} points remaining.")
                 # Reset progress counter if pre-filter did something useful
                 iterations_since_last_progress = 0 
             else:
                 print("Pre-filter: No points found within tolerance.")
        except Exception as e:
             print(f"Error during pre-filtering: {e}. Skipping.")
             pre_covered_indices = set() 
             
    if not uncovered_indices:
        print("All points pre-filtered or dataset empty.")
        return [] 

    # --- Build KD-Tree ---
    print("Building KD-Tree...")
    kdtree = KDTree(all_points) 
    print("KD-Tree built.")

    # --- Adaptive Fitting Loop ---
    iteration_count = 0
    discarded_split = 0
    discarded_failed_fit = 0
    start_time = time.time()
    initial_uncovered_count = len(uncovered_indices)
    print(f"Starting adaptive fitting on {initial_uncovered_count} points. Max iterations w/o progress: {max_iterations_without_progress}")

    while uncovered_indices:
        iteration_count += 1
        iterations_since_last_progress += 1 # Increment progress counter

        # --- Check for Lack of Progress ---
        if iterations_since_last_progress > max_iterations_without_progress:
            print(f"\nWarning: Exceeded {max_iterations_without_progress} iterations without covering new points. Stopping.")
            break
            
        try:
             seed_idx = random.choice(list(uncovered_indices)) 
        except IndexError: 
             break # Safety break if set becomes empty unexpectedly
             
        seed_point = all_points[seed_idx]
        min_points_for_fit = n_neighbors + 1
        distances, indices = kdtree.query(seed_point, k=min_points_for_fit)
        
        if len(indices) < min_points_for_fit:
             # Don't discard seed, just try another one next iteration
             continue 

        local_points = all_points[indices]
        
        # --- Attempt Polynomial Fit ---
        coeffs, poly_transformer = find_implicit_polynomial(local_points, poly_degree)

        # --- Check Fit Success ---
        if coeffs is not None and poly_transformer is not None:
            # --- Sign Consistency Check ---
            is_splitting = False # Default to not splitting
            percent_pos = 0.0
            percent_neg = 0.0
            try:
                poly_values_local = poly_transformer.transform(local_points) @ coeffs
                count_pos = np.sum(poly_values_local > sign_check_tolerance)
                count_neg = np.sum(poly_values_local < -sign_check_tolerance)
                num_local = local_points.shape[0]
                if num_local > 0:
                    percent_pos = count_pos / num_local
                    percent_neg = count_neg / num_local
                    is_splitting = max(percent_pos, percent_neg) < min_side_percentage
                else: 
                    is_splitting = True # Cannot check consistency on empty set
            except Exception as e:
                # print(f"Error during sign check for seed {seed_idx}: {e}.") # Reduce noise
                is_splitting = True # Treat error as failure

            if is_splitting:
                discarded_split += 1
                # --- Do NOT discard seed_idx ---
                continue # Try a different seed next iteration
            
            # --- Proceed with Global Coverage Check (if fit OK and not splitting) ---
            current_uncovered_list = list(uncovered_indices) 
            if not current_uncovered_list: break 
            points_to_check = all_points[current_uncovered_list]
            
            try:
                poly_values_global = poly_transformer.transform(points_to_check) @ coeffs
                abs_poly_values_global = np.abs(poly_values_global)
                local_mask_covered = abs_poly_values_global <= tolerance
                
                # Find the indices *within the current_uncovered_list* that are covered
                covered_local_indices = np.where(local_mask_covered)[0]
                
                # Map back to original indices ONLY IF they were in uncovered_indices
                original_indices_covered_this_step = {
                    current_uncovered_list[i] for i in covered_local_indices
                }
                
                # Check if *any* points were newly covered 
                if original_indices_covered_this_step: # Progress made!
                    iterations_since_last_progress = 0 # Reset progress counter
                    
                    # Store the fit
                    initial_polynomial_fits.append({ 
                        'seed_index': seed_idx,
                        'neighbor_indices': indices[1:].tolist() if np.isclose(distances[0], 0, atol=1e-8) else indices.tolist(), # Store neighbors used
                        'coefficients': coeffs,
                        'poly_features': poly_transformer,
                        'covered_indices': list(original_indices_covered_this_step) 
                    })
                    # Update uncovered set
                    uncovered_indices.difference_update(original_indices_covered_this_step) 
                    newly_covered_count = len(original_indices_covered_this_step)

                    if (iteration_count % 100 == 0) or (newly_covered_count > initial_uncovered_count * 0.01): 
                         print(f"  Iter {iteration_count} (Prog: {iterations_since_last_progress}): Seed {seed_idx}. Fit OK. Covered {newly_covered_count}. Rem: {len(uncovered_indices)}/{initial_uncovered_count}")
                
                # else: # Fit was valid, but covered no *new* points
                    # Do nothing, loop continues, iterations_since_last_progress increases
                    # print(f"  Iter {iteration_count} (Prog: {iterations_since_last_progress}): Seed {seed_idx}. Fit OK, but no new points covered.") # Optional debug
                    pass

            except Exception as e:
                 # Error during global coverage check
                 # print(f"Error during global coverage check for seed {seed_idx}: {e}.") # Reduce noise
                 # Do not discard seed, just let progress counter increase and try again
                 pass
                 
        else: # Fit failed (find_implicit_polynomial returned None)
            discarded_failed_fit += 1
            # --- Do NOT discard seed_idx ---
            continue # Try a different seed next iteration

    # --- End of While Loop ---
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"\nFinished adaptive fitting phase in {iteration_count} iterations ({total_runtime:.2f} seconds).")
    if iterations_since_last_progress > max_iterations_without_progress:
         print(f"Stopped due to exceeding max iterations ({max_iterations_without_progress}) without progress.")
    print(f"Generated {len(initial_polynomial_fits)} initial fits ({discarded_split} discarded as splitting, {discarded_failed_fit} failed fits during phase).")
    
    # --- Post-Filtering Step ---
    if not initial_polynomial_fits:
         print("No initial polynomials were generated.")
         return []
         
    print(f"Filtering polynomials: Keeping fits covering >= {min_points_covered_threshold} points globally.")
    filtered_polynomial_fits = [
        fit for fit in initial_polynomial_fits 
        if len(fit['covered_indices']) >= min_points_covered_threshold
    ]
    num_filtered_out = len(initial_polynomial_fits) - len(filtered_polynomial_fits)
    print(f"Filtered out {num_filtered_out} polynomials based on global coverage count.")
    print(f"Returning {len(filtered_polynomial_fits)} polynomials.")

    # --- Print Kept Polynomials ---
    if filtered_polynomial_fits:
        print("\n--- Kept Polynomial Equations (Sorted by Global Coverage) ---")
        filtered_polynomial_fits.sort(key=lambda x: len(x['covered_indices']), reverse=True)
        
        for i, fit in enumerate(filtered_polynomial_fits):
             poly_str = format_polynomial(fit['coefficients'], fit['poly_features'])
             print(f"Kept Poly {i+1} (Seed: {fit['seed_index']}, Global Covered: {len(fit['covered_indices'])} points):")
             print(f"  {poly_str}")
        print("---------------------------------------------------------")
    else:
        print("\nNo polynomials met the final coverage threshold.")

    return filtered_polynomial_fits

# --- Example Usage ---
JSON_FILE = 'matrix_eigenvalue_data_filtered.json' 
N_NEIGHBORS = 2000               
POLY_DEGREE = 4                
TOLERANCE = 0.2                
PREFILTER = True               
RANDOM_SEED = 47 
MIN_COVERAGE = 5 
MIN_SIDE_PERC = 0.15 
SIGN_TOL = 0.09 
# Set max iterations w/o progress (e.g., 500, or None for default N*2)
MAX_NO_PROGRESS_ITER = 1000 


# Run the estimation with the progress check
final_polynomials = adaptive_polynomial_boundary_estimation(
    json_filepath=JSON_FILE,
    n_neighbors=N_NEIGHBORS,
    poly_degree=POLY_DEGREE,
    tolerance=TOLERANCE,
    prefilter_poly=PREFILTER,
    random_seed=RANDOM_SEED, 
    min_points_covered_threshold=MIN_COVERAGE,
    min_side_percentage=MIN_SIDE_PERC,         
    sign_check_tolerance=SIGN_TOL,              
    max_iterations_without_progress=MAX_NO_PROGRESS_ITER # Pass new parameter
)

# --- Final Coverage Check --- 
# (Coverage check logic remains the same, checking against final_polynomials and pre_covered_indices)
print("\n--- Final Coverage Check (Using Kept Polynomials) ---")
original_points = load_eigenvalue_data(JSON_FILE)
# (Rest of coverage check code...)
if original_points is not None:
    num_total_points = original_points.shape[0]
    print(f"Total points in original dataset: {num_total_points}")
    covered_by_kept_adaptive = set()
    if final_polynomials: 
        for fit in final_polynomials:
            if isinstance(fit.get('covered_indices'), list):
                 covered_by_kept_adaptive.update(fit['covered_indices'])
            # else: print(f"Warning: 'covered_indices' missing/invalid for fit seeded at {fit.get('seed_index')}")
    print(f"Total unique points covered by KEPT adaptive polynomials: {len(covered_by_kept_adaptive)}")
    pre_covered_indices = set()
    if PREFILTER:
         try:
             p0_values = 1.0 + np.sum(original_points[:, :4], axis=1)
             pre_covered_mask = np.abs(p0_values) <= TOLERANCE
             pre_covered_indices = set(np.where(pre_covered_mask)[0])
             print(f"Points covered by pre-filter (rechecked): {len(pre_covered_indices)}")
         except Exception: pre_covered_indices = set() 
    total_covered_final = len(covered_by_kept_adaptive.union(pre_covered_indices))
    print(f"Total unique points covered (pre-filter + KEPT adaptive): {total_covered_final}")
    num_uncovered_final = num_total_points - total_covered_final
    if num_uncovered_final == 0:
        print("Coverage check: OK. All points covered.")
    else:
        print(f"Coverage check: {num_uncovered_final} points are NOT covered by the final set of polynomials.")
elif not final_polynomials:
     print("\nNo polynomials were kept after filtering.")
     # (Check pre-filter coverage as before if needed)