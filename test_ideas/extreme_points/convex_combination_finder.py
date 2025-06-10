import numpy as np
from scipy.optimize import linprog

def find_convex_combination_coefficients(target_point_tuple, other_points_tuples_list, 
                                       point_labels=None, display_tolerance=1e-7):
    """
    Finds and displays the convex combination coefficients for a target point
    using a list of other points.

    Args:
        target_point_tuple (tuple): The d-dimensional point to be expressed.
        other_points_tuples_list (list of tuples): List of d-dimensional points
                                                   to form the combination.
        point_labels (list of str, optional): Labels for the other_points_tuples_list.
                                              If None, default labels P0, P1, ... are used.
        display_tolerance (float): Coefficients smaller than this will not be individually
                                   displayed but are part of the sum.
    """
    target_np = np.array(target_point_tuple, dtype=float)
    other_points_np_list = [np.array(p, dtype=float) for p in other_points_tuples_list]
    
    num_other_points = len(other_points_np_list)
    if num_other_points == 0:
        print("Error: No points provided in 'other_points_tuples_list'.")
        return None
        
    dimension = len(target_np)
    if not all(p.shape == (dimension,) for p in other_points_np_list):
        print("Error: Target point and all other points must have the same dimension.")
        return None

    # A_eq setup:
    # First 'dimension' rows: coefficients of lambda_i for each coordinate
    # Last row: sum of lambdas = 1
    A_coords_matrix = np.array(other_points_np_list).T  # Shape: dimension x num_other_points
    A_sum_of_lambdas_row = np.ones((1, num_other_points))      
    A_eq = np.vstack([A_coords_matrix, A_sum_of_lambdas_row]) # Shape: (dimension + 1) x num_other_points
    
    # b_eq setup:
    # First 'dimension' elements: target point coordinates
    # Last element: 1 (for sum of lambdas)
    b_coords_vector = target_np                         
    b_sum_of_lambdas_val = np.array([1.0])                  
    b_eq = np.concatenate([b_coords_vector, b_sum_of_lambdas_val]) 

    # Objective function: minimize 0 (we only need a feasible solution)
    c_objective = np.zeros(num_other_points)
    
    # Bounds for each lambda_i: 0 <= lambda_i <= 1
    lambda_bounds = [(0, 1) for _ in range(num_other_points)]
    
    # Solve the linear program
    # 'highs' is the default and recommended method in recent SciPy versions.
    result = linprog(c_objective, A_eq=A_eq, b_eq=b_eq, bounds=lambda_bounds, method='highs')

    if result.success:
        lambdas = result.x
        
        # Verification (using a slightly larger tolerance for floating point checks)
        verification_tol = 1e-6 
        reconstructed_point = np.dot(lambdas, np.array(other_points_np_list))
        sum_of_lambdas = np.sum(lambdas)
        
        # Check if the solution is valid within tolerance
        valid_reconstruction = np.allclose(reconstructed_point, target_np, atol=verification_tol)
        valid_sum = np.isclose(sum_of_lambdas, 1.0, atol=verification_tol)
        non_negative_lambdas = np.all(lambdas >= -verification_tol) # Allow very small negatives

        if not (valid_reconstruction and valid_sum and non_negative_lambdas):
            print("Warning: Linprog reported success, but solution verification failed critical checks.")
            print(f"  Target Point: {target_np.tolist()}")
            print(f"  Reconstructed by linprog: {reconstructed_point.tolist()}")
            print(f"  Sum of lambdas from linprog: {sum_of_lambdas:.8f}")
            print(f"  Lambdas from linprog: {[f'{l:.6f}' for l in lambdas]}")
            # Continue to display coefficients but with caution

        print(f"\nConvex combination found for P = {target_point_tuple}:")
        
        contributing_points_details = []
        if point_labels is None:
            point_labels = [f"P{i}" for i in range(num_other_points)]
            
        for i, lambda_val in enumerate(lambdas):
            # Ensure lambda_val is treated as zero if it's extremely small negative
            effective_lambda = max(0, lambda_val)
            if effective_lambda > display_tolerance: 
                print(f"  {effective_lambda:.6f} * {point_labels[i]} {other_points_tuples_list[i]}")
                contributing_points_details.append((point_labels[i], other_points_tuples_list[i], effective_lambda))
        
        if not contributing_points_details:
             print("  Note: All coefficients are very small or zero (within display tolerance).")
             # Check if the target is identical to one of the 'other' points
             for i, p_other_tuple in enumerate(other_points_tuples_list):
                 if np.allclose(np.array(p_other_tuple), target_np, atol=1e-9):
                     print(f"  Target point {target_np.tolist()} is virtually identical to {point_labels[i]} {p_other_tuple}.")
                     print(f"  The combination is: 1.0 * {point_labels[i]}")
                     return {point_labels[i]: 1.0} # Return the trivial combination

        print(f"\nVerification of the combination:")
        print(f"  Target Point P                  = {target_np.tolist()}")
        print(f"  Calculated Sum(lambda_i * P_i)  = {reconstructed_point.tolist()}")
        print(f"  Sum of all lambda_i             = {sum_of_lambdas:.8f}")
        
        return {label: coeff for label, _, coeff in contributing_points_details}
        
    else:
        print(f"\nCould not find a convex combination for P = {target_point_tuple}.")
        print(f"  Linear programming solver status: {result.message}")
        if result.status == 1: print("  Solver may have reached iteration limit.")
        elif result.status == 2: print("  Problem declared infeasible (point may not be in the convex hull).")
        elif result.status == 3: print("  Problem declared unbounded (should not occur here).")
        elif result.status == 4: print("  Numerical difficulties encountered.")
        return None

if __name__ == '__main__':
    # Define point P5 using cosine values
    cos_2pi_5 = np.cos(2 * np.pi / 5)
    cos_4pi_5 = np.cos(4 * np.pi / 5)

    # Full list of points as tuples
    all_points = [
        (1.0, 1.0, 1.0, 1.0),                          # P0
        (1.0, 1.0, 1.0, -1.0),                         # P1
        (1.0, 1.0, -1.0, -1.0),                        # P2
        (1.0, 1.0, -0.5, -0.5),                        # P3
        (1.0, -0.5, -0.5, -1.0),                       # P4
        (cos_2pi_5, cos_2pi_5, cos_4pi_5, cos_4pi_5),  # P5
        (-0.25, -0.25, -0.25, -0.25),                  # P6
        (1.0, -0.5, -0.5, -0.5),                       # P7
        (0.0, 0.0, 0.0, -0.8),                         # P8
        (0.8, 0.0, 0.0, -0.2)                          # P9 (This is the target point)
    ]

    target_P_tuple = all_points[9]
    other_Ps_tuples = all_points[:9] # Points P0 through P8
    
    # Create labels P0, P1, ..., P8 for the other points
    other_Ps_labels = [f"P{i}" for i in range(len(other_Ps_tuples))]

    print(f"Attempting to express point P_target = {target_P_tuple}")
    print(f"as a convex combination of the following {len(other_Ps_tuples)} points:")
    for i, p_tuple in enumerate(other_Ps_tuples):
        print(f"  {other_Ps_labels[i]}: {p_tuple}")

    # Find and display the combination
    # Using a display_tolerance of 1e-7 to show coefficients that are meaningfully non-zero.
    combination_coeffs = find_convex_combination_coefficients(
        target_P_tuple, 
        other_Ps_tuples, 
        point_labels=other_Ps_labels,
        display_tolerance=1e-7 
    )

    if combination_coeffs:
        print("\n--- Final Summary of Coefficients ---")
        # Reconstruct for final clean print
        reconstructed_final = np.zeros(len(target_P_tuple), dtype=float)
        sum_coeffs_final = 0.0
        for label, coeff_val in combination_coeffs.items():
            # Find original point tuple from label
            original_point_idx = int(label[1:]) # Assumes P0, P1, etc.
            original_point_tuple = other_Ps_tuples[original_point_idx]
            print(f"  {coeff_val:.6f} * {label} {original_point_tuple}")
            reconstructed_final += coeff_val * np.array(original_point_tuple, dtype=float)
            sum_coeffs_final += coeff_val
        print("--------------------------------------")
        print(f"  Final Reconstructed Point  = {reconstructed_final.tolist()}")
        print(f"  Original Target Point      = {list(target_P_tuple)}")
        print(f"  Sum of Displayed Coeffs  = {sum_coeffs_final:.8f}")