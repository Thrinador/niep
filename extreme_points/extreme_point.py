#
# -----------------------------------------------------------------------------
#
# CONVEX HULL ANALYSIS SCRIPT
#
# -----------------------------------------------------------------------------
#
# Description:
#   This script performs a comprehensive analysis of a set of points in relation
#   to their convex hull. This version includes a final correction to the
#   facet sorting logic to ensure it is purely numerical.
#
# Workflow:
#   1.  Analysis of Hull Defining Points: Iterates through the initial points,
#       reporting if each is NECESSARY (a vertex) or REDUNDANT (internal).
#       Redundant points are immediately expressed as a convex combination of
#       the necessary vertices.
#   2.  External Point Identification: Finds points from an external JSON file
#       that lie outside the convex hull.
#   3.  Furthest Point Reporting: Ranks and reports the top N furthest
#       external points.
#   4.  Exposed Facet Characterization: Using only the necessary vertices, it
#       identifies all exposed facets of the hull, sorts the facets themselves
#       into a canonical numerical order, and prints their descriptions.
#
# -----------------------------------------------------------------------------

import numpy as np
import json
import os
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError
from scipy.optimize import linprog
from collections import defaultdict

HULL_DEFINING_POINTS = [
    (1, 1, 1, 1, 1),
    (1, 1, 1, 1, -1),
    (1, 1, 1, -1, -1),
    (1, 1, -1, -1, -1),
    (-0.2, -0.2, -0.2, -0.2, -0.2),
    (1,-0.5,-0.5,-0.5, -0.5),
    (0,0,0,0,-1),
    (1,0,0,-1,-1),
    (1,-1/3,-1/3,-1/3,-1),
    (0.25, 0.25, 0, -0.75, -0.75),
    (1, 0.25, 0.25, -0.75, -0.75)
]
POINTS_TO_CHECK_JSON_PATH = '../sub_sniep/data/sub_sniep_n6_dims11_11_11_11.json'
NUM_FURTHEST_POINTS_TO_DISPLAY = 5
TOLERANCE = 1e-5


def classify_hull_points(points_list):
    """
    Silently checks each point in a list to see if it's a necessary vertex.
    """
    if not points_list: return [], {}
    try:
        dimension = len(points_list[0])
        points_np_list = [np.array(p, dtype=float) for p in points_list]
    except (TypeError, IndexError): return [], {}

    necessary_points, redundant_points_info = [], {}
    for i, current_point_tuple in enumerate(points_list):
        current_point_to_check = points_np_list[i]
        other_points_np = [p for j, p in enumerate(points_np_list) if j != i]

        if len(other_points_np) < dimension + 1:
            necessary_points.append(current_point_tuple)
            continue
        try:
            hull_of_others = ConvexHull(np.array(other_points_np), qhull_options='QJ')
        except QhullError:
            necessary_points.append(current_point_tuple)
            continue
        
        signed_distances = np.dot(hull_of_others.equations[:, :-1], current_point_to_check) + hull_of_others.equations[:, -1]
        if np.all(signed_distances <= TOLERANCE):
            redundant_points_info[current_point_tuple] = None
        else:
            necessary_points.append(current_point_tuple)
    return necessary_points, redundant_points_info

def find_furthest_external_points(json_file_path, hull_points_list, num_furthest):
    """
    Identifies and ranks points from a JSON file that are outside a given convex hull.
    """
    print(f"Searching for the {num_furthest} furthest points outside the hull...")
    print(f"Using input file: {json_file_path}")
    try:
        hull = ConvexHull(np.array(hull_points_list), qhull_options='QJ')
    except (QhullError, ValueError) as e:
        print(f"   Error: Could not construct the convex hull. {e}")
        return []

    external_points = []
    try:
        with open(json_file_path, 'r') as f: data_from_file = json.load(f)
    except FileNotFoundError:
        print(f"   Error: Input JSON file not found at '{json_file_path}'"); return []
    except json.JSONDecodeError:
        print(f"   Error: Could not decode JSON from '{json_file_path}'"); return []

    if not isinstance(data_from_file, list):
        print("   Warning: JSON content is not a list."); return []

    for entry in data_from_file:
        if isinstance(entry, dict) and 'eigenvalues' in entry and len(entry['eigenvalues']) == hull.ndim:
            point_np = np.array(entry['eigenvalues'])
            signed_distances = np.dot(hull.equations[:, :-1], point_np) + hull.equations[:, -1]
            if np.any(signed_distances > TOLERANCE):
                external_points.append((np.max(signed_distances), entry))

    if not external_points:
        print("   No points from the JSON file were found outside the convex hull."); return []

    external_points.sort(key=lambda x: x[0], reverse=True)
    return external_points[:num_furthest]

def characterize_exposed_facets(hull_points_list, point_labels):
    """
    Computes, SORTS, and describes the unique exposed facets of the convex hull.
    """
    try:
        hull = ConvexHull(np.array(hull_points_list), qhull_options='QJ')
    except (QhullError, ValueError) as e:
        print(f"   Error: Could not compute hull to find facets. {e}"); return []
        
    # 1. Gather details for all unique facets first
    facet_details, processed_keys = [], set()
    for i in range(len(hull.simplices)):
        canonical_key = tuple(sorted(hull.simplices[i]))
        if canonical_key in processed_keys: continue
        processed_keys.add(canonical_key)
        
        coeffs, offset = hull.equations[i, :-1], hull.equations[i, -1]
        
        vertex_labels_unsorted = [point_labels[tuple(hull_points_list[idx])] for idx in canonical_key]
        vertex_labels_sorted = sorted(vertex_labels_unsorted, key=lambda label: int(label[1:]))
        
        # --- THIS IS THE KEY FIX ---
        # Create a purely numerical key for sorting the facets themselves
        numerical_sort_key = [int(label[1:]) for label in vertex_labels_sorted]
        
        facet_details.append({
            'numerical_sort_key': numerical_sort_key, # Use this to sort the facets
            'equation_coeffs': coeffs,
            'equation_offset': offset,
            'vertices': vertex_labels_sorted
        })

    # 2. Sort the facets based on their numerical vertex list
    facet_details.sort(key=lambda item: item['numerical_sort_key'])

    # 3. Now, build the final formatted description strings in the sorted order
    final_descriptions = []
    decimals = 5
    for i, detail in enumerate(facet_details):
        facet_counter = i + 1
        coeffs, offset = detail['equation_coeffs'], detail['equation_offset']
        
        eq_parts = [f"{c:+.{decimals-2}f}*x{j+1}" for j, c in enumerate(coeffs) if not np.isclose(c, 0)]
        eq_str = " + ".join(eq_parts).replace("+ -", "- ")
        
        desc = (f"- Facet {facet_counter}:\n"
                f"  - Equation (approx): {eq_str} <= {-offset:.{decimals}f}\n"
                f"  - Vertices: {{{', '.join(detail['vertices'])}}}")
        final_descriptions.append(desc)
        
    return final_descriptions

def express_as_convex_combination(target_point, basis_points, basis_labels):
    """
    Finds and displays the convex combination coefficients for a target point.
    """
    target_np, basis_points_np = np.array(target_point), np.array(basis_points)
    A_eq = np.vstack([basis_points_np.T, np.ones((1, len(basis_points)))])
    b_eq = np.concatenate([target_np, [1.0]])
    res = linprog(c=np.zeros(len(basis_points)), A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')

    if res.success:
        print("     └─ Convex Combination Found:")
        reconstructed = np.zeros_like(target_np, dtype=float)
        for i, l_val in enumerate(res.x):
            if l_val > TOLERANCE:
                point = basis_points[i]
                label = basis_labels[point]
                print(f"         {l_val:.6f} * {label} {point}")
                reconstructed += l_val * np.array(point)
        print("\n       Verification:",
              f"\n         - Sum of Coefficients: {np.sum(res.x):.6f}",
              f"\n         - Reconstructed Point: {tuple(np.round(reconstructed, 6))}",
              f"\n         - Original Target:     {target_point}\n")
    else:
        print(f"     └─ Failed to find a convex combination. Solver status: {res.message}\n")


if __name__ == '__main__':
    print("=====================================================")
    print("      Convex Hull Analysis Starting")
    print("=====================================================")

    # --- STEP 1: Analysis of Hull Defining Points ---
    necessary_points, redundant_points_info = classify_hull_points(HULL_DEFINING_POINTS)

    if not necessary_points or len(necessary_points) < len(HULL_DEFINING_POINTS[0]) + 1:
        print("\nAnalysis HALTED: Not enough necessary points found to define a valid hull.")
        quit()

    print("\n--- [Step 1: Analysis of Hull Defining Points] ---")
    necessary_point_labels = {pt: f"P{i+1}" for i, pt in enumerate(necessary_points)}
    
    print("Necessary points are:")
    for point in HULL_DEFINING_POINTS:
        if not point in redundant_points_info:
            label = necessary_point_labels.get(point, "??")
            print(f"-> Point {label} {point}")
    print()
    print ("Redundant points are:")
    for point in HULL_DEFINING_POINTS:
        if point in redundant_points_info:
            print(f"-> Point {point}")
            express_as_convex_combination(point, necessary_points, necessary_point_labels)

    # --- STEP 2: Finding External Points ---
    print("\n--- [Step 2: Finding External Points from JSON] ---")
    furthest_points = find_furthest_external_points(
        POINTS_TO_CHECK_JSON_PATH, necessary_points, NUM_FURTHEST_POINTS_TO_DISPLAY
    )
    if furthest_points:
        print(f"\nTop {len(furthest_points)} furthest points found:")
        for i, (dist, data) in enumerate(furthest_points):
            print(f"  {i+1}. Distance = {dist:.6f}, Point Eigenvalues = {data['eigenvalues']}")
    
    # --- STEP 3: Characterizing the Exposed Hull Facets ---
    print("\n--- [Step 3: Characterizing the Exposed Hull Facets] ---")
    facet_descriptions = characterize_exposed_facets(necessary_points, necessary_point_labels)
    if facet_descriptions:
        print(f"\nFound {len(facet_descriptions)} unique facets:")
        for desc in facet_descriptions: print(desc)
    else:
            print("\nCould not characterize hull facets.")

    print("\n=====================================================")
    print("      Convex Hull Analysis Complete")
    print("=====================================================")