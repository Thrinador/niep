import json
import numpy as np
import plotly.graph_objects as go
import sys # For exiting if file not found

def plot_filtered_eigenvalues(json_file_path):
    """
    Loads filtered data from a JSON file and creates a 3D scatter plot
    of its eigenvalues using Plotly.
    Eigenvalues are sorted in descending order: x=λ1 (largest), y=λ2, z=λ3, color=λ4 (smallest).
    Tooltip includes all four eigenvalues.
    """
    try:
        with open(json_file_path, 'r') as f:
            filtered_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at '{json_file_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'")
        sys.exit(1)

    if not isinstance(filtered_data, list):
        print("Error: JSON data is not a list of items.")
        sys.exit(1)

    print(f"Loaded {len(filtered_data)} data points from '{json_file_path}'.")

    e1_largest_list, e2_list, e3_list, e4_smallest_list = [], [], [], []
    valid_items_count = 0

    for item_idx, item in enumerate(filtered_data):
        if not isinstance(item, dict) or "eigenvalues" not in item:
            continue
        
        eigenvalues_raw = item["eigenvalues"]
        if not isinstance(eigenvalues_raw, list) or len(eigenvalues_raw) != 4:
            continue
            
        try:
            sorted_eigs_asc = np.sort(np.array(eigenvalues_raw, dtype=float))
            if len(sorted_eigs_asc) == 4:
                e1_largest_list.append(sorted_eigs_asc[3]) 
                e2_list.append(sorted_eigs_asc[2])         
                e3_list.append(sorted_eigs_asc[1])         
                e4_smallest_list.append(sorted_eigs_asc[0])
                valid_items_count += 1
        except ValueError:
            continue

    if valid_items_count == 0:
        print("No valid eigenvalue data found to plot.")
        return

    print(f"Plotting {valid_items_count} valid data points...")

    # Define the hover template
    # %{x}, %{y}, %{z} refer to the x, y, z coordinates of the point
    # %{marker.color} refers to the value used for coloring the marker (our λ₄)
    # :.4f formats the number to 4 decimal places
    # <extra></extra> removes the secondary box with trace information
    hover_text_template = (
        "<b>λ₁ (x)</b>: %{x:.4f}<br>"
        "<b>λ₂ (y)</b>: %{y:.4f}<br>"
        "<b>λ₃ (z)</b>: %{z:.4f}<br>"
        "<b>w (λ₄)</b>: %{marker.color:.4f}"
        "<extra></extra>"  # This removes the trace info box
    )

    # Create the 3D scatter plot trace
    trace = go.Scatter3d(
        x=e1_largest_list,
        y=e2_list,
        z=e3_list,
        mode='markers',
        marker=dict(
            size=4,
            color=e4_smallest_list,
            colorscale='Viridis', 
            colorbar=dict(
                title='Smallest Eigenvalue (λ₄)'
            ),
            opacity=0.7
        ),
        hovertemplate=hover_text_template # Apply the custom hover template
    )

    # Define the layout
    layout = go.Layout(
        title_text='3D Scatter Plot of Filtered Eigenvalues (Ordered Largest to Smallest)',
        scene=dict(
            xaxis_title_text='Largest Eigenvalue (λ₁)', 
            yaxis_title_text='2nd Largest Eigenvalue (λ₂)',
            zaxis_title_text='3rd Largest Eigenvalue (λ₃)' 
        ),
        margin=dict(l=10, r=10, b=10, t=50)
    )

    # Create and show the figure
    fig = go.Figure(data=[trace], layout=layout)
    print("Displaying plot... (This may open in your web browser)")
    fig.show()

# --- Main Execution Block ---
if __name__ == '__main__':
    # Define the path to your filtered JSON data file
    # This should be the output file from the previous filtering script
    FILTERED_JSON_PATH = 'points_outside_hull.json' 

    plot_filtered_eigenvalues(FILTERED_JSON_PATH)