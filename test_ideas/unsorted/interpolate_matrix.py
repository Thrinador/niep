import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import svd

def construct_new_matrix_M(a, b, c):
    """Constructs the new 5x5 matrix M for given parameters a, b, c."""
    
    M = np.zeros((5, 5))

    M[0,0] = 1 - a - b
    M[0,1] = a
    M[0,2] = b

    M[1,0] = a
    M[1,1] = 1 - 2*a - 2*c
    M[1,2] = a
    M[1,3] = c
    M[1,4] = c

    M[2,0] = b
    M[2,1] = a
    M[2,2] = 1 - a - b

    M[3,1] = c
    M[3,3] = 1 - c
    
    M[4,1] = c
    M[4,4] = 1 - c
    
    return M

def generate_eigenvalue_samples_new_matrix(num_samples):
    """
    Generates samples of the 4 eigenvalues of interest from the new 5x5 matrix.
    Parameters a, b, c are sampled such that all matrix entries are non-negative.
    The sampling strategy is:
      0 <= c <= 0.5
      0 <= a <= 0.5 - c
      0 <= b <= 1 - a
    """
    samples = []
    attempts = 0
    max_attempts_factor = 10 
    absolute_max_attempts = max(num_samples * max_attempts_factor, num_samples + 100)

    while len(samples) < num_samples and attempts < absolute_max_attempts:
        attempts += 1
        
        c_sample = np.random.uniform(0, 0.5)
        a_upper_bound = max(0, 0.5 - c_sample) 
        a_sample = np.random.uniform(0, a_upper_bound)
        b_upper_bound = max(0, 1 - a_sample)
        b_sample = np.random.uniform(0, b_upper_bound)
        
        try:
            current_M = construct_new_matrix_M(a_sample, b_sample, c_sample)
            all_5_eigenvalues = np.linalg.eigh(current_M)[0]
            
            idx_of_1 = np.argmin(np.abs(all_5_eigenvalues - 1.0))
            four_eigenvalues_of_interest = np.delete(all_5_eigenvalues, idx_of_1)
            four_eigenvalues_of_interest.sort()

            samples.append(four_eigenvalues_of_interest)
        except Exception as e:
            if attempts <= 5 or attempts % (absolute_max_attempts // 20 + 1) == 0 : 
                 print(f"Warning (Attempt {attempts}): Error generating sample with a={a_sample:.4f}, b={b_sample:.4f}, c={c_sample:.4f}. Error: {e}")
    
    if len(samples) < num_samples:
         print(f"Warning: Only {len(samples)} samples were successfully generated out of {num_samples} requested after {attempts} attempts.")
            
    return np.array(samples)

def find_polynomial_coefficients(eigenvalue_samples, degree):
    """
    Finds the coefficients of an implicit polynomial P(l1, l2, l3, l4) = 0.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True) 
    V = poly.fit_transform(eigenvalue_samples)
    u, s, vh = svd(V)
    coefficients = vh[-1, :]
    feature_names = poly.get_feature_names_out(['l1', 'l2', 'l3', 'l4'])
    
    return coefficients, feature_names, s

# --- Main execution ---
if __name__ == '__main__':
    a_test, b_test, c_test = 0.1, 0.1, 0.05 
    print(f"Test parameters: a={a_test}, b={b_test}, c={c_test}")
    
    valid_test_params = (c_test >= 0 and c_test <= 0.5 and
                         a_test >= 0 and a_test <= 0.5 - c_test and
                         b_test >= 0 and b_test <= 1 - a_test)
    print(f"Test parameters are within the non-negative region: {valid_test_params}")
    assert valid_test_params, "Test parameters do not satisfy non-negativity constraints!"

    M_test = construct_new_matrix_M(a_test, b_test, c_test)
    print("Constructed test matrix M:\n", M_test)
    
    all_eigenvalues_test = np.linalg.eigh(M_test)[0]
    all_eigenvalues_test.sort()
    print(f"\nAll 5 eigenvalues of M_test: {all_eigenvalues_test}")
    
    assert np.any(np.isclose(all_eigenvalues_test, 1.0)), "One eigenvalue should be 1."
    
    idx_of_1_test = np.argmin(np.abs(all_eigenvalues_test - 1.0))
    four_eigenvalues_test = np.delete(all_eigenvalues_test, idx_of_1_test)
    four_eigenvalues_test.sort()
    print(f"The 4 eigenvalues of interest for M_test: {four_eigenvalues_test}")

    num_data_samples = 2000
    
    print(f"\nGenerating {num_data_samples} samples for the 4 eigenvalues of interest...")
    print("Parameters a,b,c sampled to ensure non-negative matrix entries based on derived constraints:")
    print("  0 <= c <= 0.5")
    print("  0 <= a <= 0.5 - c")
    print("  0 <= b <= 1 - a")
    eigenvalue_data = generate_eigenvalue_samples_new_matrix(num_data_samples)
    
    if eigenvalue_data.shape[0] < num_data_samples:
        print(f"Note: {eigenvalue_data.shape[0]} samples were generated out of {num_data_samples} requested.")
    
    if eigenvalue_data.shape[0] == 0:
        print("No eigenvalue samples generated. Exiting.")
    else:
        print(f"Shape of eigenvalue data: {eigenvalue_data.shape}")
        poly_degree = 5
        
        print(f"\nFitting a polynomial of degree {poly_degree}...")
        # These are the original coefficients from SVD
        original_coeffs, original_names, singular_values = find_polynomial_coefficients(eigenvalue_data, poly_degree)
        
        # New filtering and scaling logic for display
        significance_ratio = 6  # Display terms >= 1/20th of the largest term

        print(f"\nPolynomial Coefficients (terms >= 1/{int(significance_ratio)} of max, scaled so smallest *kept* term's magnitude is ~1.0):")

        if original_coeffs is None or len(original_coeffs) == 0:
            print("No coefficients computed.")
        else:
            max_abs_coeff = np.max(np.abs(original_coeffs))

            if np.isclose(max_abs_coeff, 0): 
                print("All computed coefficients are (close to) zero.")
                # Optionally, print the zero coefficients:
                # for name, coeff_val in zip(original_names, original_coeffs):
                #    print(f"{name}: {coeff_val:.6f}")
            else:
                filter_threshold = max_abs_coeff / significance_ratio
                
                kept_coeffs_data = []
                for name, coeff_val in zip(original_names, original_coeffs):
                    if np.abs(coeff_val) >= filter_threshold:
                        kept_coeffs_data.append({'name': name, 'original_coeff': coeff_val})
                
                if not kept_coeffs_data:
                    print(f"No coefficients met the relative filter threshold (abs(coeff) >= {filter_threshold:.4e}).")
                    print("Showing all original computed coefficients instead:")
                    for name, coeff_val in zip(original_names, original_coeffs):
                        print(f"{name}: {coeff_val:.6f}")
                else:
                    # Find the smallest absolute value among the kept coefficients
                    min_abs_kept_coeff = min(np.abs(item['original_coeff']) for item in kept_coeffs_data)

                    if np.isclose(min_abs_kept_coeff, 0):
                        # This case should be rare if filter_threshold > 0
                        print("Smallest kept coefficient is (close to) zero, cannot scale to make it 1. Displaying kept unscaled coefficients:")
                        for item in kept_coeffs_data:
                            print(f"{item['name']}: {item['original_coeff']:.6f}")
                    else:
                        scaling_factor = 1.0 / min_abs_kept_coeff
                        print(f"(Info: Original max_abs_coeff: {max_abs_coeff:.4e}, filter_thresh: {filter_threshold:.4e}, min_abs_kept_coeff: {min_abs_kept_coeff:.4e}, display_scaling_factor: {scaling_factor:.4e})")

                        for item in kept_coeffs_data:
                            scaled_coeff = item['original_coeff'] * scaling_factor
                            print(f"{item['name']}: {scaled_coeff:.6f}")
        
        print(f"\nSingular values of the monomial matrix V (smallest one should be close to 0 for a good fit):")
        if singular_values is not None:
            # Display more singular values to see the drop-off
            num_s_to_show = min(len(singular_values), 10) 
            print(f"Smallest {num_s_to_show} singular values (sorted ascending): {np.sort(singular_values)[:num_s_to_show]}")
            if len(singular_values) > 1:
                 print(f"Ratio of smallest to second smallest singular value: {singular_values[-1]/singular_values[-2] if len(singular_values)>1 and not np.isclose(singular_values[-2],0) else 'N/A'}")


        # Verification uses the ORIGINAL coefficients
        if original_coeffs is not None:
            poly_transform = PolynomialFeatures(degree=poly_degree, include_bias=True)
            V_data = poly_transform.fit_transform(eigenvalue_data)
            poly_values = V_data @ original_coeffs # Use original_coeffs for verification
            
            print(f"\nVerification using original (unscaled, unfiltered) coefficients:")
            print(f"Max absolute value of P(lambdas) over the samples: {np.max(np.abs(poly_values)):.2e}")
            print(f"Mean absolute value of P(lambdas) over the samples: {np.mean(np.abs(poly_values)):.2e}")
            print(f"(These values should be very close to 0 for a good polynomial fit)")
        else:
            print("\nVerification step skipped as coefficients were not computed.")