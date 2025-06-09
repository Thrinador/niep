import json
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans 
import time 
from math import comb 
import os 

# Assume load_eigenvalue_data and find_implicit_polynomial functions 
# are defined as previously. They remain unchanged.
# Re-adding them here for completeness.

def load_eigenvalue_data(json_filepath):
    """Loads 4D eigenvalue data from a JSON file."""
    try:
        with open(json_filepath, 'r') as f: data = json.load(f)
        eigenvalue_points = []
        for item in data:
            eigenvals = item.get('eigenvalues')
            if isinstance(eigenvals, (list, tuple)) and len(eigenvals) == 4:
                try:
                    float_eigenvals = [float(val) for val in eigenvals]; eigenvalue_points.append(float_eigenvals)
                except (ValueError, TypeError): pass 
        if not eigenvalue_points: print("Error: No valid 4D eigenvalue data found."); return None
        return np.array(eigenvalue_points, dtype=float) 
    except FileNotFoundError: print(f"Error: JSON file not found at {json_filepath}"); return None
    except json.JSONDecodeError: print(f"Error: Could not decode JSON from {json_filepath}"); return None
    except Exception as e: print(f"An unexpected error occurred during loading: {e}"); return None

def find_implicit_polynomial(points, degree):
    """Finds coefficients of an implicit polynomial P(x)=0 fit to a set of points."""
    if points is None or points.shape[0] < 1: return None, None 
    if points.ndim != 2 or points.shape[1] != 4: return None, None
    num_points = points.shape[0]; num_coeffs_theor = comb(degree + 4, 4) 
    if num_points < num_coeffs_theor: pass # Allow underdetermined
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    try:
        A = poly.fit_transform(points); _, s, Vh = np.linalg.svd(A)
        coefficients = Vh[-1, :]; coefficients /= np.linalg.norm(coefficients) 
        return coefficients, poly
    except np.linalg.LinAlgError: return None, None
    except Exception: return None, None

# --- Updated format_polynomial function ---
def format_polynomial(coefficients, poly_transformer, variables=['x1', 'x2', 'x3', 'x4'], 
                      precision=4, min_abs_coeff=0.05): # Changed threshold parameter
    """
    Formats a polynomial equation string, omitting terms with small coefficients.

    Args:
        coefficients (np.ndarray): The polynomial coefficients.
        poly_transformer (PolynomialFeatures): The fitted transformer.
        variables (list): Names for the variables.
        precision (int): Decimal precision for printed coefficients.
        min_abs_coeff (float): Minimum absolute value for a coefficient to be included.
                               Defaults to 0.05.

    Returns:
        str: The formatted polynomial equation string P(x) = 0.
    """
    if coefficients is None: return "Invalid Polynomial (coefficients are None)" 
    if not hasattr(poly_transformer, 'n_features_in_'): return "Invalid Polynomial (transformer not fitted)"
        
    try: feature_names = poly_transformer.get_feature_names_out(variables)
    except Exception: return "Error getting feature names"
        
    terms = []
    # --- Use min_abs_coeff for filtering ---
    zero_threshold = 1e-9 # Keep a very small threshold for strict zero check if needed
    for coeff, name in zip(coefficients, feature_names):
        abs_coeff = abs(coeff)
        # Skip term if coefficient magnitude is below the threshold
        if abs_coeff < min_abs_coeff: 
            continue
            
        sign = "-" if coeff < 0 else "+"
        
        # Handle coefficient part (1.0 vs others)
        # Use zero_threshold for closeness check to 1.0
        if np.isclose(abs_coeff, 1.0, atol=zero_threshold): 
            coeff_str = ""
            if name == '1': coeff_str = f"{1.0:.{precision}f}" 
        else: 
            coeff_str = f"{abs_coeff:.{precision}f}"
            
        # Handle variable part
        if name == '1': 
            term_str = f"{coeff_str}"
        else: 
            mult_sign = "*" if coeff_str else "" 
            term_str = f"{coeff_str}{mult_sign}{name}"
            
        terms.append((sign, term_str))

    if not terms:
        # If all terms were filtered out, mention the threshold
        return f"0 = 0 (All coefficients below threshold {min_abs_coeff})" 
        
    first_sign, first_term = terms[0]
    poly_str = f"{'-' if first_sign == '-' else ''}{first_term}"
    for sign, term in terms[1:]: poly_str += f" {sign} {term}"
    return f"{poly_str} = 0"

# --- Updated Main Function with Threshold Parameter ---

def find_boundary_curves_via_clustering(json_filepath, 
                                        n_clusters=2, 
                                        poly_degree=3, 
                                        random_seed=None,
                                        use_scaling=True,
                                        prefilter_poly=True, 
                                        prefilter_tolerance=0.1,
                                        print_coeff_threshold=0.05): # New parameter for printing
    """
    Finds dominant boundary curves via clustering, optionally pre-filtering points,
    and prints polynomials omitting small terms.

    Args:
        json_filepath (str): Path to the JSON file.
        n_clusters (int): Number of clusters to find.
        poly_degree (int): Degree of the polynomial to fit to each cluster.
        random_seed (int, optional): Seed for K-Means reproducibility.
        use_scaling (bool): Scale data before clustering.
        prefilter_poly (bool): If True, pre-filter points close to 1+x1+...+x4=0.
        prefilter_tolerance (float): Tolerance |P(x)| for pre-filtering.
        print_coeff_threshold (float): Min absolute coefficient to print in output polynomials.

    Returns:
        list: List of polynomial fit dictionaries for each cluster.
    """
    # 1. Load Data
    all_points = load_eigenvalue_data(json_filepath)
    if all_points is None or all_points.shape[0] == 0: return []
    num_total_points, num_dims = all_points.shape
    if num_dims != 4: print(f"Error: Expected 4D points, got {num_dims}D."); return []
    print(f"Loaded {num_total_points} points.")

    # 2. Optional Pre-filtering Step
    points_for_processing = all_points 
    if prefilter_poly:
        print(f"Applying pre-filter (Tolerance: {prefilter_tolerance})...")
        try:
             p0_values = 1.0 + np.sum(all_points, axis=1)
             keep_mask = np.abs(p0_values) > prefilter_tolerance
             points_for_processing = all_points[keep_mask]
             num_kept = points_for_processing.shape[0]
             num_filtered_out = num_total_points - num_kept
             if num_filtered_out > 0: print(f"Pre-filter: Removed {num_filtered_out} points. {num_kept} remaining.")
             else: print("Pre-filter: No points removed.")
             if num_kept == 0: print("Error: Pre-filter removed all points."); return []
        except Exception as e: print(f"Error during pre-filtering: {e}. Skipping."); points_for_processing = all_points

    num_processing_points = points_for_processing.shape[0]
    if num_processing_points < n_clusters: print(f"Error: Points after filtering ({num_processing_points}) < n_clusters ({n_clusters})."); return []

    # 3. Scale Data (Optional)
    scaled_points_for_processing = points_for_processing
    if use_scaling:
        print("Scaling data before clustering...")
        scaler = StandardScaler()
        try: scaled_points_for_processing = scaler.fit_transform(points_for_processing)
        except ValueError as e: print(f"Warning: Scaling failed ({e}). Using unscaled data."); scaled_points_for_processing = points_for_processing.copy()

    # 4. Perform Clustering (K-Means)
    print(f"Performing K-Means (K={n_clusters}) on {num_processing_points} points...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10) 
    try: cluster_labels = kmeans.fit_predict(scaled_points_for_processing) 
    except Exception as e: print(f"Error during K-Means: {e}"); return []
    print("Clustering complete. Label counts:", np.unique(cluster_labels, return_counts=True))

    # 5. Fit Polynomial to Each Cluster
    polynomial_fits = []
    print(f"\nFitting polynomial of degree {poly_degree} to each cluster...")
    for cluster_id in range(n_clusters):
        print(f"-- Processing Cluster {cluster_id} --")
        cluster_mask_in_filtered = (cluster_labels == cluster_id)
        cluster_points = points_for_processing[cluster_mask_in_filtered]
        num_points_in_cluster = cluster_points.shape[0]
        if num_points_in_cluster == 0: print("  Cluster empty. Skipping."); continue
        print(f"  Found {num_points_in_cluster} points.")
        num_coeffs_needed = comb(poly_degree + 4, 4)
        if num_points_in_cluster < num_coeffs_needed: print(f"  Warning: {num_points_in_cluster} points < {num_coeffs_needed} coeffs needed. Skipping."); continue

        coeffs, poly_transformer = find_implicit_polynomial(cluster_points, poly_degree)
        if coeffs is not None and poly_transformer is not None:
            print(f"  Successfully fitted polynomial.")
            polynomial_fits.append({'cluster_id': cluster_id, 'num_points': num_points_in_cluster, 'coefficients': coeffs, 'poly_features': poly_transformer})
        else: print(f"  Failed to fit polynomial for this cluster.")

    # 6. Print and Return Results
    if polynomial_fits:
        print(f"\n--- Found Polynomial Equations (Terms with |coeff| >= {print_coeff_threshold}) ---")
        polynomial_fits.sort(key=lambda x: x['cluster_id']) 
        for i, fit in enumerate(polynomial_fits):
             # Pass the print threshold down to format_polynomial
             poly_str = format_polynomial(fit['coefficients'], 
                                          fit['poly_features'], 
                                          min_abs_coeff=print_coeff_threshold) 
             print(f"Curve {i+1} (From Cluster {fit['cluster_id']}, {fit['num_points']} points):")
             print(f"  {poly_str}")
        print("-------------------------------------------------------------")
    else:
        print("\nNo successful polynomial fits were obtained from the clusters.")

    return polynomial_fits


# --- Example Usage ---
JSON_FILE = 'matrix_eigenvalue_data_filtered.json' 
N_CLUSTERS = 1   
POLY_DEGREE = 3   
RANDOM_SEED = 43 
USE_SCALING = True 
PREFILTER = True 
PREFILTER_TOL = 0.1 
PRINT_COEFF_THRESHOLD = 0.05 # Define the print threshold


# Run the clustering-based estimation with print threshold
found_curves = find_boundary_curves_via_clustering(
    json_filepath=JSON_FILE,
    n_clusters=N_CLUSTERS,
    poly_degree=POLY_DEGREE,
    random_seed=RANDOM_SEED,
    use_scaling=USE_SCALING,
    prefilter_poly=PREFILTER,          
    prefilter_tolerance=PREFILTER_TOL,  
    print_coeff_threshold=PRINT_COEFF_THRESHOLD # Pass new threshold parameter
)

if not found_curves:
    print("\nFailed to find boundary curves using the clustering method.")
    # (Suggestions...)