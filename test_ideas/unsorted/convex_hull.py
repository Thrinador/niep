import json
import numpy as np
from scipy.spatial import ConvexHull, KDTree

def reduce_similar_points(points, tolerance):
    """
    Reduces a point cloud by keeping only one representative point from
    clusters of points that are within a given tolerance of each other.

    Args:
        points (numpy.ndarray): An array of points (N_points, N_dimensions).
        tolerance (float): The distance threshold. Points within this distance
                           of a selected representative are considered part of its cluster.

    Returns:
        numpy.ndarray: A reduced array of points.
    """
    if points.shape[0] == 0:
        return np.array([])

    # Build a KD-tree for efficient nearest neighbor search
    tree = KDTree(points)
    
    processed_mask = np.zeros(points.shape[0], dtype=bool)
    unique_points_list = []

    for i in range(points.shape[0]):
        if not processed_mask[i]:
            # This point is a new representative
            unique_points_list.append(points[i])
            processed_mask[i] = True
            
            # Find all points within 'tolerance' of points[i]
            # query_ball_point returns a list of lists of indices.
            # We are querying for a single point, so we take the first (and only) list.
            neighbor_indices = tree.query_ball_point(points[i], r=tolerance, return_sorted=True)
            
            for idx in neighbor_indices:
                processed_mask[idx] = True
                
    return np.array(unique_points_list)

def find_convex_hull_from_eigenvalues(data, point_reduction_tolerance=1e-5):
    """
    Extracts 4D points from 'eigenvalues', reduces similar points,
    and computes the convex hull.

    Args:
        data (list): List of dictionaries with 'eigenvalues'.
        point_reduction_tolerance (float): Tolerance for reducing similar points.
                                           Set to 0 or None to disable reduction.

    Returns:
        tuple: (all_extracted_points, reduced_points, hull_vertices_points, hull_object)
    """
    eigenvalue_points = []
    for entry in data:
        if "eigenvalues" in entry:
            if isinstance(entry["eigenvalues"], list) and len(entry["eigenvalues"]) == 4:
                eigenvalue_points.append(entry["eigenvalues"])
            else:
                print(f"Warning: 'eigenvalues' field in entry is not a list of 4 numbers. Entry: {entry}")
        else:
            print(f"Warning: Entry missing 'eigenvalues' field. Entry: {entry}")

    if not eigenvalue_points:
        print("Error: No valid 4D eigenvalue points found in the data.")
        return np.array([]), np.array([]), None, None

    all_extracted_points = np.array(eigenvalue_points)
    print(f"Total number of 4D points extracted: {all_extracted_points.shape[0]}")

    # --- Point Reduction Step ---
    if point_reduction_tolerance is not None and point_reduction_tolerance > 0:
        print(f"Reducing similar points with tolerance: {point_reduction_tolerance}...")
        reduced_points = reduce_similar_points(all_extracted_points, tolerance=point_reduction_tolerance)
        print(f"Number of points after reduction: {reduced_points.shape[0]}")
    else:
        print("Skipping point reduction.")
        reduced_points = all_extracted_points
    
    current_points_for_hull = reduced_points

    if current_points_for_hull.shape[0] == 0:
        print("Error: No points remaining after reduction (if applied).")
        return all_extracted_points, current_points_for_hull, None, None

    # Need N+1 points for N-D hull (e.g., 5 points for 4D)
    if current_points_for_hull.shape[0] <= current_points_for_hull.shape[1]:
        print(f"Error: Not enough unique points ({current_points_for_hull.shape[0]}) to form a convex hull in {current_points_for_hull.shape[1]}D.")
        print(f"You need at least {current_points_for_hull.shape[1] + 1} points in {current_points_for_hull.shape[1]}D to define a volume.")
        return all_extracted_points, current_points_for_hull, None, None

    try:
        hull = ConvexHull(current_points_for_hull)
        hull_vertices_points = current_points_for_hull[hull.vertices]
        return all_extracted_points, current_points_for_hull, hull_vertices_points, hull
    except Exception as e:
        print(f"Error computing convex hull: {e}")
        return all_extracted_points, current_points_for_hull, None, None

# --- Main script execution ---
file_path = 'sniep/data/ds-sniep_n5_dims15_15_15.json'
data_objects = []

# Load data from the specified JSON file
try:
    with open(file_path, 'r') as f:
        data_objects = json.load(f)
    print(f"Successfully loaded data from {file_path}")
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from the file {file_path}. Please check the file format.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the file: {e}")
    exit()

if not isinstance(data_objects, list):
    print(f"Error: The JSON file {file_path} does not contain a top-level array (list) of objects.")
    exit()
if not data_objects:
    print(f"Warning: The file {file_path} was loaded but is empty or contains no processable data.")
    exit()

# --- Configuration for Point Reduction ---
# Adjust this tolerance based on your data's characteristics.
# A smaller value means points must be very close to be merged.
# If set to None or 0, point reduction is skipped.
SIMILARITY_TOLERANCE = 8e-1  # Example: points within 0.00001 distance are considered similar

# Find the convex hull using the loaded data
(all_original_points,
 processed_points,
 generating_hull_points,
 convex_hull_object) = find_convex_hull_from_eigenvalues(data_objects, 
                                                      point_reduction_tolerance=SIMILARITY_TOLERANCE)

if generating_hull_points is not None:
    print(f"\nNumber of points used for Convex Hull computation (after reduction): {len(processed_points)}")
    print(f"Number of generating points for the 4D convex hull: {len(generating_hull_points)}")
    print("Generating points of the 4D convex hull:")
    for i, point_coords in enumerate(generating_hull_points):
        print(f"  Vertex {i}: {point_coords}")
else:
    if processed_points.size > 0:
        print("\nCould not compute convex hull with the processed points.")
        print("This might be due to not enough unique points after reduction,")
        print("points being degenerate (e.g., all co-planar in a lower dimension), or other numerical issues.")
    elif all_original_points.size > 0 and processed_points.size == 0 :
         print("\nAll points were filtered out by the reduction process. Try a smaller tolerance.")
    # Other error messages are handled within find_convex_hull_from_eigenvalues or during file loading.