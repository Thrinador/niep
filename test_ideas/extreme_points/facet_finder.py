import numpy as np
from scipy.spatial import ConvexHull
from collections import defaultdict

# Define the original 5D points
points_5d = np.array([
    [1, 1, 1, 1, 1],  # p1 (index 0)
    [1, 1, 1, 1, -1], # p2 (index 1)
    [1, 1, 1, -1, -1],# p3 (index 2)
    [1, 1, -1/2, -1/2, -1/2], # p4 (index 3)
    [1, 1, -1/2, -1/2, -1], # p5 (index 4)
    [1, np.cos(2*np.pi/5), np.cos(2*np.pi/5), np.cos(4*np.pi/5), np.cos(4*np.pi/5)], # p6 (index 5)
    [1, 0, 0, 0, -0.8], # p7 (index 6)
    [1, -1/4, -1/4, -1/4, -1/4] # p8 (index 7)
])

# Project points to 4D
points_4d = points_5d[:, 1:]

try:
    hull = ConvexHull(points_4d)
except Exception as e:
    print(f"Error computing ConvexHull: {e}")
    hull = None

# To store unique facets: key is canonical equation, value is set of vertex indices
unique_facets_vertices = defaultdict(set)
unique_facets_equations = {} # To store one representative equation string for each canonical key

DECIMALS = 5 # For rounding to identify unique equations

if hull:
    for i in range(len(hull.simplices)):
        simplex_indices = hull.simplices[i] # Indices of vertices forming this simplicial part of a facet
        equation_params = hull.equations[i] # Normal vector and offset for this simplex

        coeffs = equation_params[:-1]
        offset = equation_params[-1]

        # Canonical form for the equation
        norm = np.linalg.norm(coeffs)
        if np.isclose(norm, 0): # Should not happen for a valid facet
            continue
        
        coeffs_normalized = coeffs / norm
        offset_normalized = offset / norm

        # Ensure a consistent sign convention (e.g., first non-zero coeff is positive)
        # Or if all normal coeffs are zero (not possible for facet normal), check offset sign
        # A simpler approach: if the first element with significant magnitude is negative, flip all.
        # Find first element with magnitude > 1e-4 (tolerance for being "zero")
        first_significant_idx = -1
        for k_idx, c_val in enumerate(coeffs_normalized):
            if abs(c_val) > 1e-4:
                first_significant_idx = k_idx
                break
        
        if first_significant_idx != -1 and coeffs_normalized[first_significant_idx] < 0:
            coeffs_normalized *= -1
            offset_normalized *= -1
        elif first_significant_idx == -1: # All normal coeffs are effectively zero, should not happen
             if offset_normalized < 0: # Arbitrary convention for this unlikely case
                offset_normalized *= -1
        
        # Create a canonical key for the dictionary
        canonical_eq_key = (tuple(np.round(coeffs_normalized, decimals=DECIMALS)),
                              np.round(offset_normalized, decimals=DECIMALS))

        # Add vertices to this unique facet
        # Vertex indices from hull.simplices are 0-based for the input 'points_4d'
        for vertex_idx in simplex_indices:
            unique_facets_vertices[canonical_eq_key].add(vertex_idx)

        # If this is the first time we see this canonical equation, store its string form
        if canonical_eq_key not in unique_facets_equations:
            eq_str_parts = []
            for j_idx in range(len(coeffs)):
                 if not np.isclose(coeffs[j_idx], 0):
                    eq_str_parts.append(f"{coeffs[j_idx]:+.{DECIMALS}f} x_{j_idx+2}")
            if not eq_str_parts and not np.isclose(offset,0): # Only offset
                 eq_str = f"0 = {-offset:.{DECIMALS}f}" # Should simplify to a contradiction unless offset is 0
            elif not eq_str_parts and np.isclose(offset,0):
                 eq_str = "0 = 0" # Degenerate
            else:
                eq_str = " + ".join(eq_str_parts).replace("+ -", "- ") + f" = {-offset:.{DECIMALS}f}"
            unique_facets_equations[canonical_eq_key] = eq_str
    
    # Output the results
    print(f"The polytope has {len(unique_facets_vertices)} unique facets.")
    facet_counter = 1
    
    # Prepare for LaTeX output by storing in a list
    global facets_data_for_user_output
    facets_data_for_user_output = []

    for canonical_key, vertex_indices_set in unique_facets_vertices.items():
        # Sort indices and convert to 1-based for p_i notation
        sorted_vertex_labels = sorted([f"p_{idx+1}" for idx in vertex_indices_set])
        
        facets_data_for_user_output.append({
            "id": facet_counter,
            "equation_str": unique_facets_equations[canonical_key],
            "vertices_list_str": ", ".join(sorted_vertex_labels)
        })
        facet_counter += 1

    # This data would then be formatted for the final response.
    # For verification here, print plain text:
    for data_item in facets_data_for_user_output:
        print(f"\nFacet {data_item['id']}:")
        print(f"  Equation (approximate): {data_item['equation_str']}")
        print(f"  Is the convex hull of points: {{{data_item['vertices_list_str']}}}")