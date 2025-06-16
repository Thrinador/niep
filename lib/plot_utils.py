import logging
import math
import plotly.graph_objects as go
import os
import json
import numpy as np
import itertools

from lib import file_utils

def filter_points_grid(points, tolerance):
    """
    Filters points using a high-performance grid-based spatial partitioning method.

    This approach is significantly faster for large datasets than the O(N^2) naive
    comparison method. Its average time complexity is closer to O(N).

    Args:
        points (list of list of floats): The list of points to filter.
        tolerance (float): The minimum distance required between points.

    Returns:
        list of list of floats: The filtered list of points.
    """
    if not points or tolerance <= 0:
        return points

    # Determine the number of dimensions from the first point
    dims = len(points[0])
    cell_size = float(tolerance)
    
    # The grid will store a mapping from cell coordinates to the points within that cell
    grid = {}
    kept_points = []

    for p in points:
        # Determine the integer coordinates of the cell the point belongs to
        cell_coords = tuple(int(coord / cell_size) for coord in p)
        
        is_too_close = False
        
        # Generate offsets to check the current cell and all its immediate neighbors.
        # For d dimensions, this checks 3^d cells in total.
        for offset in itertools.product([-1, 0, 1], repeat=dims):
            neighbor_cell_coords = tuple(cell_coords[i] + offset[i] for i in range(dims))
            
            if neighbor_cell_coords in grid:
                # Check distance against all points already kept in this neighboring cell
                for other_p in grid[neighbor_cell_coords]:
                    # Using numpy for fast Euclidean distance calculation
                    dist = np.linalg.norm(np.array(p) - np.array(other_p))
                    if dist < tolerance:
                        is_too_close = True
                        break # Found a conflict
            if is_too_close:
                break # Move to the next point
        
        if not is_too_close:
            # If no conflicts were found, keep the point
            kept_points.append(p)
            # Add the new point to the grid for subsequent checks
            if cell_coords not in grid:
                grid[cell_coords] = []
            grid[cell_coords].append(p)
            
    return kept_points

def run_plotting(config):
    n = config['global_data']['n']
    
    # The plot_tolerance defines the minimum distance between points.
    plot_tolerance = config.get('plot_tolerance', 1e-2)
    # The number of frames to generate for the animation.
    num_animation_frames = config.get('num_animation_frames', 30)


    if n < 3 or n > 5:
        logging.error(f"Plot size of n={n} is not supported! Current options are n=3,4,5")
        exit()

    plot_dir = file_utils.build_file_name_no_extension(config, 'plots')

    # 2. Load and extract the data
    data_filename = file_utils.build_file_name(config)
    if not os.path.exists(data_filename): 
        logging.error(f"Data file not found: {data_filename}")
        exit()

    logging.info(f"Loading data from: {data_filename}")
    try:
        with open(data_filename, 'r') as f: 
            data = json.load(f)
        logging.info(f"Loaded {len(data)} entries.")
    except Exception as e: 
        logging.error(f"Failed to load data: {e}")
        exit()
    if not data: 
        logging.warning("Loaded data empty.")
        exit()

    all_coeffs, all_eigs = [], []
    logging.info("Extracting data...")
    for entry in data:
        all_coeffs.append(entry.get("coefficients") if isinstance(entry.get("coefficients"), list) else None)
        all_eigs.append(entry.get("eigenvalues") if isinstance(entry.get("eigenvalues"), list) else None)
    logging.info("Finished extracting data.")

    try: 
        os.makedirs(plot_dir, exist_ok=True)
        logging.info(f"Ensured plot directory exists: {plot_dir}")
    except OSError as e: 
        logging.error(f"Could not create plot dir: {e}")
        exit()

    create_plots(n, plot_dir, "coefficients", all_coeffs, plot_tolerance, num_animation_frames)
    create_plots(n, plot_dir, "eigenvalues", all_eigs, plot_tolerance, num_animation_frames)

    print("Plot generation finished.")

def create_plots_helper_2d(run_identifier, plot_type, all_points, i, j, tolerance):
    logging.info(f"Generating {plot_type} 2d plot for indicies {i+1}, {j+1}")
    
    points_to_plot = [[p[i], p[j]] for p in all_points]
    
    filtered_points = filter_points_grid(points_to_plot, tolerance)
    logging.info(f"Filtered {len(points_to_plot)} points down to {len(filtered_points)}.")

    if not filtered_points:
        logging.warning("All points were filtered out. Skipping plot generation.")
        return

    coeffs_x, coeffs_y = zip(*filtered_points)

    fig_coeffs = go.Figure()
    marker_config = dict(size=5)

    fig_coeffs.add_trace(go.Scatter(x=list(coeffs_x), y=list(coeffs_y), mode='markers', marker=marker_config))
    fig_coeffs.update_layout(title=f'{plot_type} space ({run_identifier}) for E_{i+1} and E_{j+1}', 
                            scene=dict(
                                xaxis_title=f"{plot_type} {i+1}", 
                                yaxis_title=f"{plot_type} {j+1}", 
                            ))

    fig_coeffs.write_html(os.path.join(run_identifier, f"{plot_type}_2d_{i+1}_{j+1}.html"), include_mathjax='cdn')

def create_plots_helper_3d(run_identifier, plot_type, all_points, i, j, k, tolerance):
    logging.info(f"Generating {plot_type} 3d plot for indicies {i+1}, {j+1}, {k+1}")
    
    points_to_plot = [[p[i], p[j], p[k]] for p in all_points]

    filtered_points = filter_points_grid(points_to_plot, tolerance)
    logging.info(f"Filtered {len(points_to_plot)} points down to {len(filtered_points)}.")
    
    if not filtered_points:
        logging.warning("All points were filtered out. Skipping plot generation.")
        return

    coeffs_x, coeffs_y, coeffs_z = zip(*filtered_points)

    fig_coeffs = go.Figure()
    marker_config = dict(size=5)

    fig_coeffs.add_trace(go.Scatter3d(x=list(coeffs_x), y=list(coeffs_y), z=list(coeffs_z), mode='markers', marker=marker_config))
    
    fig_coeffs.update_layout(title=f'{plot_type} Space ({run_identifier}) for E_{i+1}, E_{j+1}, and E_{k+1}', 
                            scene=dict(
                                xaxis_title=f"{plot_type} {i+1}", 
                                yaxis_title=f"{plot_type} {j+1}", 
                                zaxis_title=f"{plot_type} {k+1}"
                            ), 
                            margin=dict(l=0, r=0, b=0, t=40))

    fig_coeffs.write_html(os.path.join(run_identifier, f"{plot_type}_3d_{i+1}_{j+1}_{k+1}.html"), include_mathjax='cdn')

def create_plots_helper_4d(run_identifier, plot_type, all_points, i, j, k, l, tolerance):
    logging.info(f"Generating {plot_type} 4d plot for indicies {i+1}, {j+1}, {k+1}, {l+1}")
    
    points_to_plot = [[p[i], p[j], p[k], p[l]] for p in all_points]

    filtered_points = filter_points_grid(points_to_plot, tolerance)
    logging.info(f"Filtered {len(points_to_plot)} points down to {len(filtered_points)}.")

    if not filtered_points:
        logging.warning("All points were filtered out. Skipping plot generation.")
        return

    coeffs_x, coeffs_y, coeffs_z, coeffs_color = zip(*filtered_points)

    fig_coeffs = go.Figure()
    marker_config = dict(size=5)

    marker_config.update(dict(color=list(coeffs_color), colorscale='Viridis', showscale=True, colorbar_title=f"{plot_type} {l+1}"))
    fig_coeffs.add_trace(go.Scatter3d(x=list(coeffs_x), y=list(coeffs_y), z=list(coeffs_z), mode='markers', marker=marker_config))
    fig_coeffs.update_layout(title=f'{plot_type} Space ({run_identifier}) for E_{i+1}, E_{j+1}, E_{k+1}, and E_{l+1}', 
                            scene=dict(
                                xaxis_title=f"{plot_type} {i+1}", 
                                yaxis_title=f"{plot_type} {j+1}", 
                                zaxis_title=f"{plot_type} {k+1}"
                            ), 
                            margin=dict(l=0, r=0, b=0, t=40))

    fig_coeffs.write_html(os.path.join(run_identifier, f"{plot_type}_4d_{i+1}_{j+1}_{k+1}_{l+1}.html"), include_mathjax='cdn')

def create_plots_helper_4d_animation(run_identifier, plot_type, all_points, i, j, k, l, tolerance, num_frames):
    logging.info(f"Generating {plot_type} 4d animation for indicies (x,y,z,t): {i+1},{j+1},{k+1},{l+1} with {num_frames} frames.")

    points_4d = np.array([[p[i], p[j], p[k], p[l]] for p in all_points])
    
    min_time = points_4d[:, 3].min()
    max_time = points_4d[:, 3].max()

    if min_time == max_time:
        logging.warning("Cannot create animation: 4th dimension (time) has no range.")
        return

    # Create the time bins for the animation frames
    bins = np.linspace(min_time, max_time, num_frames + 1)
    
    fig = go.Figure()

    # Determine global axis ranges from all points to keep axes constant
    axis_range = dict(
        xaxis=dict(range=[points_4d[:, 0].min(), points_4d[:, 0].max()], autorange=False),
        yaxis=dict(range=[points_4d[:, 1].min(), points_4d[:, 1].max()], autorange=False),
        zaxis=dict(range=[points_4d[:, 2].min(), points_4d[:, 2].max()], autorange=False)
    )

    # Create a frame for each time bin
    frames = []
    for f_idx in range(num_frames):
        t_start = bins[f_idx]
        t_end = bins[f_idx+1]

        # Select points within the current time bin
        if f_idx == num_frames - 1:
            mask = (points_4d[:, 3] >= t_start) & (points_4d[:, 3] <= t_end)
        else:
            mask = (points_4d[:, 3] >= t_start) & (points_4d[:, 3] < t_end)
        
        frame_points_unfiltered = points_4d[mask]

        # Filter points within the frame if any exist
        if frame_points_unfiltered.shape[0] > 0:
            points_to_filter = frame_points_unfiltered[:, :3].tolist()
            filtered_3d_points = filter_points_grid(points_to_filter, tolerance)
        else:
            filtered_3d_points = []
        
        if not filtered_3d_points:
            frame_x, frame_y, frame_z = [], [], []
        else:
            frame_x, frame_y, frame_z = zip(*filtered_3d_points)

        # The first frame's data becomes the initial trace
        if f_idx == 0:
            fig.add_trace(go.Scatter3d(
                x=list(frame_x), y=list(frame_y), z=list(frame_z),
                mode='markers', marker=dict(size=5)
            ))

        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=list(frame_x), y=list(frame_y), z=list(frame_z),
                mode='markers', marker=dict(size=5)
            )],
            name=f"{t_start:.3f}" # Name the frame by the start of its time bin
        ))
    
    fig.frames = frames

    # Configure animation settings
    def create_animation_settings(duration):
        return {
            "frame": {"duration": duration, "redraw": True},
            "transition": {"duration": 0},
            "fromcurrent": True,
            "mode": "immediate"
        }

    sliders = [{
        "steps": [
            {"args": [[f.name], create_animation_settings(0)], "label": f.name, "method": "animate"}
            for f in fig.frames
        ],
        "active": 0, "transition": {"duration": 0},
        "x": 0.1, "len": 0.9, "xanchor": "left", "y": 0, "yanchor": "top"
    }]

    fig.update_layout(
        title=f'Animated {plot_type} Space ({run_identifier})<br>Axes: {i+1}, {j+1}, {k+1} | Time: {l+1}',
        scene=axis_range,
        updatemenus=[{
            "buttons": [
                {"args": [None, create_animation_settings(50)], "label": "Play", "method": "animate"},
                {"args": [[None], create_animation_settings(0)], "label": "Pause", "method": "animate"}
            ],
            "direction": "left", "pad": {"r": 10, "t": 70}, "type": "buttons", 
            "x": 0.1, "y": 0, "yanchor": "top", "xanchor": "right"
        }],
        sliders=sliders
    )
    
    # Save to HTML
    output_filename = os.path.join(run_identifier, f"{plot_type}_animation_4d_{i+1}_{j+1}_{k+1}_vs_{l+1}.html")
    fig.write_html(output_filename, include_mathjax='cdn')
    logging.info(f"Saved animation to {output_filename}")


def create_plots(n, run_identifier, plot_type, all_points, tolerance, num_animation_frames):
    logging.info(f"Generating {plot_type} plots with tolerance {tolerance}...")

    if not all_points or not all_points[0]:
        logging.warning(f"No points data provided for {plot_type}. Skipping plot generation.")
        return

    num_dimensions = len(all_points[0])

    # Make all the 2d plots
    if n >= 3 and num_dimensions >= 2:
        for i in range(num_dimensions):
            for j in range(i + 1, num_dimensions):
                create_plots_helper_2d(run_identifier, plot_type, all_points, i, j, tolerance)

    # Make all the 3d plots
    if n >= 4 and num_dimensions >= 3:
        for i in range(num_dimensions):
            for j in range(i + 1, num_dimensions):
                for k in range(j + 1, num_dimensions):
                    create_plots_helper_3d(run_identifier, plot_type, all_points, i, j, k, tolerance)

    # Make all the 4d plots
    if n >= 5 and num_dimensions >= 4:
        for i in range(num_dimensions):
            for j in range(i + 1, num_dimensions):
                for k in range(j + 1, num_dimensions):
                    for l in range(k + 1, num_dimensions):
                        create_plots_helper_4d(run_identifier, plot_type, all_points, i, j, k, l, tolerance)
                        create_plots_helper_4d_animation(run_identifier, plot_type, all_points, i, j, k, l, tolerance, num_animation_frames)
