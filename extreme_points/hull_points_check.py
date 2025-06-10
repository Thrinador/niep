import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError # For specific error catching

def check_necessary_extreme_points(points_list, tolerance=1e-9):
    """
    Checks each point in a list to see if it's a necessary extreme point
    for the convex hull of that list of points.

    Args:
        points_list (list of tuples/lists): The list of d-dimensional points.
        tolerance (float): A small positive value for checking if a point is
                           inside a hull, accounting for floating-point inaccuracies.
                           Points are considered redundant if they are within this
                           tolerance of the hull formed by other points.

    Returns:
        dict: A dictionary where keys are string representations of the input points
              and values are booleans (True if the point is necessary, False if redundant).
              Returns an empty dict or error message if input is problematic.
    """
    if not points_list:
        print("Input points list is empty.")
        return {}

    num_total_points = len(points_list)
    
    try:
        # Assuming all points have the same dimension, get from the first point
        if num_total_points == 0: # Should be caught by 'if not points_list' but defensive
             print("Input points list is empty after initial check.")
             return {}
        dimension = len(points_list[0])
        if dimension <= 0:
            print(f"Error: Point dimension ({dimension}) must be positive.")
            return {f"Error: Invalid point dimension {dimension}": False}

        # Convert points to NumPy arrays for numerical operations
        points_np_list = [np.array(p, dtype=float) for p in points_list]

        # Validate point dimensions
        for i, p_np in enumerate(points_np_list):
            if p_np.shape != (dimension,):
                print(f"Error: Point {i+1} {points_list[i]} has incorrect dimension {p_np.shape}, expected ({dimension},).")
                return {f"Error: Dimension mismatch for point {points_list[i]}": False}

    except (TypeError, IndexError) as e:
        print(f"Error processing input points (ensure list of lists/tuples of numbers): {e}")
        return {f"Error: Invalid point data structure or content": False}

    results = {} # To store results: point_str -> is_necessary (bool)

    for i in range(num_total_points):
        current_point_to_check = points_np_list[i]
        # Create a list of other points (all points except current_point_to_check)
        other_points_for_hull = [points_np_list[j] for j in range(num_total_points) if j != i]
        
        # Store point as a string for dictionary key and readable output
        point_str = f"Point {i+1} {tuple(points_list[i])}"

        # To form a d-dimensional hull, we need at least d+1 points.
        if len(other_points_for_hull) < dimension + 1:
            results[point_str] = True # Point is necessary if others are too few
            print(f"  {point_str} is NECESSARY (too few other points to form a {dimension}D hull).")
            continue

        other_points_array = np.array(other_points_for_hull)

        try:
            # Attempt to compute the convex hull of the 'other_points_for_hull'
            # 'QJ' (joggle) option helps Qhull handle precision issues or nearly co-planar points
            hull_of_others = ConvexHull(other_points_array, qhull_options='QJ')
        except QhullError:
            # If ConvexHull fails for the subset (e.g., they become degenerate, co-planar in a lower dim),
            # it implies the current_point_to_check was critical for the full hull's structure/dimension.
            results[point_str] = True
            print(f"  {point_str} is NECESSARY (ConvexHull of other points failed, likely due to degeneracy).")
            continue
        except Exception as e: # Catch other potential errors from ConvexHull
            results[point_str] = True # Treat as necessary if sub-hull computation fails unexpectedly
            print(f"  {point_str} is NECESSARY (Unexpected error computing ConvexHull of other points: {e}).")
            continue

        # Now, check if current_point_to_check is inside the hull_of_others
        # hull_of_others.equations provides [normal_vector_components..., offset]
        # For a point x, and an equation [A, b] (where A is normal, b is offset),
        # the point is inside if A.x + b <= 0 for all equations.
        # We use A.x + b <= tolerance to account for floating point issues.
        facet_normals = hull_of_others.equations[:, :-1]
        facet_offsets = hull_of_others.equations[:, -1]
        
        signed_distances = np.dot(facet_normals, current_point_to_check) + facet_offsets
        
        # If the point is inside or on the boundary (within tolerance) of hull_of_others, it's redundant.
        if np.all(signed_distances <= tolerance):
            results[point_str] = False # Redundant
            print(f"  {point_str} is REDUNDANT.")
        else:
            results[point_str] = True # Necessary
            print(f"  {point_str} is NECESSARY.")
            
    return results

if __name__ == '__main__':
    # Calculate cosine values for one of the points
    cos_2pi_5 = np.cos(2 * np.pi / 5)  # approx 0.309016994
    cos_4pi_5 = np.cos(4 * np.pi / 5)  # approx -0.809016994

    # Your updated list of 4D points
    user_points = [
        (1, 1, 1, 1), 
        (1, 1, 1, -1), 
        (1, 1, -1, -1),
        (1, -0.5, -0.5, -1),
        (cos_2pi_5, cos_2pi_5, cos_4pi_5, cos_4pi_5),
        (-0.25, -0.25, -0.25, -0.25), 
        (1,-0.5,-0.5,-0.5), 
        (0, 0, 0, -1),
        (1,0,-1,-1),
        (0.5,-0.5,-0.5,-0.5)
    ]
    
    print("Checking for necessary extreme points (vertices) in the provided list:")
    print("--------------------------------------------------------------------")
    
    # A small positive tolerance is used for checking point inclusion.
    # This helps manage floating-point inaccuracies.
    # Adjust if your data has specific precision characteristics.
    results_summary = check_necessary_extreme_points(user_points, tolerance=1e-9)
    
    print("\n--- Summary of Results ---")
    if not results_summary or any("Error" in key for key in results_summary):
        print("Could not complete the analysis due to errors in input or processing.")
    else:
        all_points_are_necessary = True
        redundant_points_found = []
        for point_description, is_point_necessary in results_summary.items():
            status_message = "NECESSARY" if is_point_necessary else "REDUNDANT"
            print(f"{point_description}: {status_message}")
            if not is_point_necessary:
                all_points_are_necessary = False
                redundant_points_found.append(point_description)
        
        print("--------------------------")
        if all_points_are_necessary:
            print("All points in the provided list are necessary extreme points for their convex hull.")
        else:
            print("Some points in the provided list are redundant:")
            for rp in redundant_points_found:
                print(f"   - {rp} is REDUNDANT.")
            print("Consider removing redundant points if a minimal set of vertices is desired for defining the hull.")