import json
import numpy as np
from scipy.spatial import ConvexHull # Changed from Delaunay
from scipy.spatial.qhull import QhullError # For specific error catching
import os

def find_points_outside_hull_desc_sorted_tolerant(json_file_path, 
                                                 hull_defining_points_list, 
                                                 output_json_path,
                                                 tolerance=1e-9): # Added tolerance parameter
    """
    Identifies points from a JSON file that lie outside the convex hull,
    considering a specified tolerance.
    The 'eigenvalues' of each point from the JSON are sorted in DESCENDING order before testing.
    Saves the original JSON objects of points found truly outside (beyond tolerance) to an output JSON file.

    Args:
        json_file_path (str): Path to the input JSON file.
        hull_defining_points_list (list of tuples): Points defining the convex hull.
        output_json_path (str): Path to save the JSON file of points outside the hull.
        tolerance (float): A small positive value. Points are considered outside
                           only if they exceed any hull hyperplane boundary by more than this tolerance.
                           Default is 1e-9.
    """
    hull_defining_points_np = np.array(hull_defining_points_list)

    try:
        # Use ConvexHull. 'QJ' option for joggling can improve robustness with degenerate input.
        hull = ConvexHull(hull_defining_points_np, qhull_options='QJ')
    except QhullError as e:
        print(f"Error creating ConvexHull (QhullError): {e}")
        print("This can happen if the points defining the hull are degenerate (e.g., co-planar).")
        return
    except Exception as e:
        print(f"An unexpected error occurred during ConvexHull creation: {e}")
        return

    # hull.equations contains [normal_vector_components..., offset]
    # For a point x, and an equation [A, b] (where A is normal, b is offset from hull.equations),
    # the point is inside if A.x + b <= 0.
    # We check A.x + b <= tolerance.
    facet_normals = hull.equations[:, :-1]
    facet_offsets = hull.equations[:, -1]

    desc_sorted_points_for_testing = []
    original_json_objects = []

    try:
        with open(json_file_path, 'r') as f:
            data_from_file = json.load(f)
            data_to_process = []
            if not isinstance(data_from_file, list):
                if isinstance(data_from_file, dict) and 'eigenvalues' in data_from_file:
                    data_to_process = [data_from_file]
                else:
                    print(f"Error: JSON file content at '{json_file_path}' is not an array of objects or a single valid object.")
                    return
            else:
                data_to_process = data_from_file
            
            for entry in data_to_process:
                if isinstance(entry, dict) and 'eigenvalues' in entry and \
                   isinstance(entry['eigenvalues'], list) and len(entry['eigenvalues']) == 5:
                    original_eigenvalues = entry['eigenvalues']
                    current_point_desc_sorted = sorted(original_eigenvalues, reverse=True)
                    desc_sorted_points_for_testing.append(current_point_desc_sorted)
                    original_json_objects.append(entry) 
                else:
                    print(f"Skipping entry due to missing, malformed, or non-4D 'eigenvalues': {str(entry)[:100]}...")
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at '{json_file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'")
        return

    if not desc_sorted_points_for_testing:
        print("No valid 4D points (eigenvalues) found in the input JSON file to check.")
        # Handle empty output file
        try:
            with open(output_json_path, 'w') as f_out: json.dump([], f_out, indent=4)
            print(f"No points to check, an empty list has been saved to '{output_json_path}'")
        except IOError: print(f"Error: Could not write empty list to output file '{output_json_path}'")
        return

    outside_objects_to_save = []
    for i, p_sorted_list in enumerate(desc_sorted_points_for_testing):
        point_np = np.array(p_sorted_list)
        
        # Calculate signed distances to each facet plane: A*x + b
        signed_distances = np.dot(facet_normals, point_np) + facet_offsets
        
        # If any distance is greater than the tolerance, the point is considered truly outside.
        if np.any(signed_distances > tolerance):
            outside_objects_to_save.append(original_json_objects[i])

    if outside_objects_to_save:
        print(f"\nFound {len(outside_objects_to_save)} JSON entries whose 'eigenvalues' (when sorted descending and checked with tolerance={tolerance}) are outside the convex hull.")
        try:
            output_dir = os.path.dirname(output_json_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir); print(f"Created output directory: '{output_dir}'")
            with open(output_json_path, 'w') as f_out: json.dump(outside_objects_to_save, f_out, indent=4)
            print(f"The original entries for these points have been saved to '{output_json_path}'")
        except IOError: print(f"Error: Could not write to output file '{output_json_path}'")
        except OSError as e: print(f"Error creating output directory for '{output_json_path}': {e}")
    else:
        print(f"\nAll JSON entries' 'eigenvalues' (when sorted descending and checked with tolerance={tolerance}) are considered inside or acceptably close to the convex hull.")
        try:
            with open(output_json_path, 'w') as f_out: json.dump([], f_out, indent=4) 
            print(f"No points were outside the hull (based on tolerance check). An empty list has been saved to '{output_json_path}'.")
        except IOError: print(f"Error: Could not write empty list to output file '{output_json_path}'")

if __name__ == '__main__':
    cos_2pi_5 = np.cos(2 * np.pi / 5)
    cos_4pi_5 = np.cos(4 * np.pi / 5)

    hull_points = [
        (1, 1, 1, 1, 1), 
        (1, 1, 1, 1, -1), 
        (1, 1, 1, -1, -1),
        (1, 1, -1, -1, -1),
        (1, 1, -0.5, -0.5, -1), 
        (1, cos_2pi_5, cos_2pi_5, cos_4pi_5, cos_4pi_5),
        (-0.2, -0.2, -0.2, -0.2, -0.2), 
        (1, 1,-0.5,-0.5,-0.5), 
        (1,-0.5,-0.5,-0.5, -0.5),
    ]

    input_filepath = '../sniep/data/sniep_n6_dims5_5_5_5.json'
    output_filepath = 'points_outside_hull.json' # Updated default name
    
    # Define your desired tolerance. 
    # A small positive value, e.g., 1e-9, 1e-7.
    # If a point is outside a facet by more than this, it's flagged.
    custom_tolerance = 0.05

    print(f"Defining convex hull with {len(set(map(tuple,hull_points)))} unique points.")
    print("Points defining the hull (these are not sorted for hull definition):")
    for p in hull_points: print(f"  {p}")
    
    print(f"\nChecking points from JSON file: '{input_filepath}'")
    print(f"Note: Each point's 'eigenvalues' will be sorted in DESCENDING order before testing.")
    print(f"A tolerance of {custom_tolerance} will be used for the 'outside' check.")
    print(f"Original JSON entries for points found truly outside will be saved to: '{output_filepath}'")

    find_points_outside_hull_desc_sorted_tolerant(
        input_filepath, 
        hull_points, 
        output_filepath,
        tolerance=custom_tolerance
    )