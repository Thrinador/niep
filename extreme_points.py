#
# -----------------------------------------------------------------------------
#
# CONVEX HULL ANALYSIS SCRIPT (CONFIG-DRIVEN)
#
# -----------------------------------------------------------------------------
#
# Description:
#   This script performs a comprehensive analysis of a set of points in relation
#   to their convex hull. All parameters are loaded from an external
#   `config.toml` file, and point sets are loaded from a specified JSON file.
#
# Workflow:
#   0.  Configuration Loading: Reads `config.toml` to get all settings.
#   1.  Point Generation: Loads the specified point set from a JSON file. If
#       `use_permutations` is True, it generates all unique permutations.
#   2.  Analysis of Hull Defining Points: Classifies every point as either
#       NECESSARY (a vertex) or REDUNDANT (internal).
#   3.  External Point Identification: Finds points from an external JSON file
#       that lie outside the convex hull.
#   4.  Furthest Point Reporting: Ranks and reports the top N furthest
#       external points.
#   5.  Exposed Facet Characterization: Describes all exposed facets of the hull.
#
# -----------------------------------------------------------------------------

import numpy as np
import json
import os
import tomli
from scipy.spatial import ConvexHull, QhullError
from scipy.optimize import linprog
from collections import defaultdict
import itertools

from lib import file_utils

# --- All configuration is now loaded from config.toml ---

def classify_hull_points(points_list, tolerance):
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
        if np.all(signed_distances <= tolerance):
            redundant_points_info[current_point_tuple] = None
        else:
            necessary_points.append(current_point_tuple)
    return necessary_points, redundant_points_info

def characterize_exposed_facets(hull_points_list, point_labels):
    """
    Computes, SORTS, and describes the unique exposed facets of the convex hull.
    """
    try:
        hull = ConvexHull(np.array(hull_points_list), qhull_options='QJ')
    except (QhullError, ValueError) as e:
        print(f"   Error: Could not compute hull to find facets. {e}"); return []
        
    facet_details, processed_keys = [], set()
    for i in range(len(hull.simplices)):
        canonical_key = tuple(sorted(hull.simplices[i]))
        if canonical_key in processed_keys: continue
        processed_keys.add(canonical_key)
        
        coeffs, offset = hull.equations[i, :-1], hull.equations[i, -1]
        
        vertex_labels_unsorted = [point_labels[tuple(hull_points_list[idx])] for idx in canonical_key]
        vertex_labels_sorted = sorted(vertex_labels_unsorted, key=lambda label: int(label[1:]))
        
        numerical_sort_key = [int(label[1:]) for label in vertex_labels_sorted]
        
        facet_details.append({
            'numerical_sort_key': numerical_sort_key,
            'equation_coeffs': coeffs,
            'equation_offset': offset,
            'vertices': vertex_labels_sorted
        })

    facet_details.sort(key=lambda item: item['numerical_sort_key'])

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

def express_as_convex_combination(target_point, basis_points, basis_labels, tolerance):
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
            if l_val > tolerance:
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


def find_furthest_external_points(input_json_path, hull_points_list, num_furthest, tolerance, output_json_path):
    """
    Identifies, saves, and ranks points from a JSON file that are outside a given convex hull.
    """
    print(f"Searching for the {num_furthest} furthest points outside the hull...")
    print(f"Using input file: {input_json_path}")
    try:
        hull = ConvexHull(np.array(hull_points_list), qhull_options='QJ')
    except (QhullError, ValueError) as e:
        print(f"   Error: Could not construct the convex hull. {e}")
        return []

    external_points_with_dist = []
    try:
        with open(input_json_path, 'r') as f: data_from_file = json.load(f)
    except FileNotFoundError:
        print(f"   Error: Input JSON file not found at '{input_json_path}'"); return []
    except json.JSONDecodeError:
        print(f"   Error: Could not decode JSON from '{input_json_path}'"); return []

    if not isinstance(data_from_file, list):
        print("   Warning: JSON content is not a list."); return []

    for entry in data_from_file:
        if isinstance(entry, dict) and 'eigenvalues' in entry and len(entry['eigenvalues']) == hull.ndim:
            point_np = np.array(entry['eigenvalues'])
            signed_distances = np.dot(hull.equations[:, :-1], point_np) + hull.equations[:, -1]
            if np.any(signed_distances > tolerance):
                external_points_with_dist.append((np.max(signed_distances), entry))

    if not external_points_with_dist:
        print("   No points from the JSON file were found outside the convex hull."); return []

    points_to_save = [entry for dist, entry in external_points_with_dist]
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(output_json_path, 'w') as f:
            json.dump(points_to_save, f, indent=4)
        print(f"   Successfully saved {len(points_to_save)} external points to '{output_json_path}'")
    except IOError as e:
        print(f"   Warning: Could not save external points to '{output_json_path}'. Error: {e}")
    
    # Sort and return only the top N furthest points for console display
    external_points_with_dist.sort(key=lambda x: x[0], reverse=True)
    return external_points_with_dist[:num_furthest]

if __name__ == '__main__':
    print("=====================================================")
    print("      Convex Hull Analysis Starting")
    print("=====================================================")

    # --- STEP 0: Load Configuration ---
    try:
        # NOTE: The library was changed to 'tomli' for consistency with Python's recommendations
        with open("config.toml", "rb") as f:
            config = tomli.load(f)
        
        extreme_points_config = config['extreme_points_data']
        use_permutations = extreme_points_config['use_permutations']
        points_to_check_path = extreme_points_config['points_to_check_path']
        
        # This logic for a dynamic path was present in your attached script
        if not points_to_check_path:
            print("Defaulting to config for points to check path")
            points_to_check_path = file_utils.build_file_name(config)

        num_furthest_points = extreme_points_config['num_furthest_points']
        tol = extreme_points_config['tolerance']
        
        hull_points_path = extreme_points_config['hull_points_path']
        hull_points_set_name = extreme_points_config['hull_points_set_name']
        if not hull_points_set_name:
            print("Defaulting to config for hull points set name")
            hull_points_set_name = f"{config['global_data']['matrix_type']}_{config['global_data']['n']}"
        external_points_output_path = extreme_points_config['external_points_output_path'] # Load new config value
        
    except (FileNotFoundError, KeyError) as e:
        print(f"FATAL: Could not load configuration from config.toml. Error: {e}")
        quit()
    except NameError as e:
        print(f"FATAL: A configuration value is likely missing or incorrect. Error: {e}")
        quit()


    # --- STEP 1: Point Generation (Ordered vs. Unordered) ---
    try:
        with open(hull_points_path, 'r') as f:
            all_point_sets = json.load(f)
        hull_defining_points = [tuple(p) for p in all_point_sets[hull_points_set_name]]
    except (FileNotFoundError, KeyError, TypeError) as e:
        print(f"FATAL: Could not load point set '{hull_points_set_name}' from '{hull_points_path}'. Error: {e}")
        quit()
        
    if use_permutations:
        print(f"Mode: Unordered (using permutations of point set '{hull_points_set_name}')")
        permuted_points_set = set()
        for point in hull_defining_points:
            perms = set(itertools.permutations(point))
            permuted_points_set.update(perms)
        analysis_points = list(permuted_points_set)
        print(f"Generated {len(analysis_points)} unique points from {len(hull_defining_points)} base points.")
    else:
        print(f"Mode: Ordered (using point set '{hull_points_set_name}' as is)")
        analysis_points = hull_defining_points
        print(f"Using {len(analysis_points)} specified points.")

    # --- STEP 2: Analysis of Hull Defining Points ---
    necessary_points, _ = classify_hull_points(analysis_points, tol)
    
    if not necessary_points or len(necessary_points) < len(analysis_points[0]) + 1:
        print("\nAnalysis HALTED: Not enough necessary points found to define a valid hull.")
        quit()

    print("\n--- [Step 2: Analysis of Hull Defining Points] ---")

    necessary_points.sort()
    necessary_point_labels = {pt: f"P{i+1}" for i, pt in enumerate(necessary_points)}
    necessary_points_set = set(necessary_points)

    print(f"Found {len(necessary_points)} necessary points (vertices of the final hull):")
    for point in necessary_points:
        label = necessary_point_labels.get(point)
        print(f"-> {label} {point}")

    all_points_set = set(map(tuple, analysis_points))
    redundant_points = sorted(list(all_points_set - necessary_points_set))
    
    print(f"\nFound {len(redundant_points)} redundant points (internal to the final hull):")
    if not redundant_points:
        print("   None")
    else:
        for point in redundant_points:
            print(f"-> Point {point}")
            express_as_convex_combination(point, necessary_points, necessary_point_labels, tol)

    # --- STEP 3: Finding External Points ---
    print("\n--- [Step 3: Finding External Points from JSON] ---")
    furthest_points = find_furthest_external_points(
        points_to_check_path, necessary_points, num_furthest_points, tol, external_points_output_path
    )
    if furthest_points:
        print(f"\nTop {len(furthest_points)} furthest points found:")
        for i, (dist, data) in enumerate(furthest_points):
            print(f"  {i+1}. Distance = {dist:.6f}, Point Eigenvalues = {data['eigenvalues']}")
    
    # --- STEP 4: Characterizing the Exposed Hull Facets ---
    print("\n--- [Step 4: Characterizing the Exposed Hull Facets] ---")
    facet_descriptions = characterize_exposed_facets(necessary_points, necessary_point_labels)
    if facet_descriptions:
        print(f"\nFound {len(facet_descriptions)} unique facets:")
        for desc in facet_descriptions: print(desc)
    else:
        print("\nCould not characterize hull facets.")

    print("\n=====================================================")
    print("      Convex Hull Analysis Complete")
    print("=====================================================")