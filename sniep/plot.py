# create_plots_cwd_config_plain_labels.py

import json
import os
import argparse
import re
import logging
import collections.abc
import time
import math # For isnan, abs

# Use tomllib if available (Python 3.11+), otherwise try tomli
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        logging.error("Please install tomli: pip install tomli")
        exit(1)

# Plotting library
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    logging.error("Please install plotly: pip install plotly")
    exit(1)


# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper to build filename (adapted from file_utils.py) ---
def build_data_filename(config):
    """Builds the consolidated data filename and the base name for outputs."""
    try:
        n = config['global_data']['n']; points_dim = config['global_data']['points_dim']
        base = "ds-sniep"; dims_str = "_".join(map(str, points_dim)); base_dir = "data"
        run_identifier = f"{base}_n{n}_dims{dims_str}"
        filename = os.path.join(base_dir, f"{run_identifier}.json")
        logging.debug(f"Generated data filename: {filename}")
        return filename, run_identifier
    except KeyError as e: logging.error(f"Config key error: {e}"); return None, None
    except Exception as e: logging.exception("Error building filename:"); return None, None

# --- Main Plotting Function ---
def create_plots(config_path):
    """Loads config, loads data, creates and saves plots into a run-specific subdir."""
    logging.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, "rb") as f: config = tomllib.load(f)
    except FileNotFoundError: logging.error(f"Config file not found: {config_path}"); return
    except Exception as e: logging.error(f"Failed to load config: {e}"); return

    plot_config = config.get('plot_data', {})
    num_slices = plot_config.get('num_eigenvalue_slices', 0)
    slice_tolerance = plot_config.get('slice_tolerance', 0.01)
    if not isinstance(num_slices, int) or num_slices < 0: num_slices = 0
    if not isinstance(slice_tolerance, (int, float)) or slice_tolerance < 0: slice_tolerance = 0.01

    data_filename, run_identifier = build_data_filename(config)
    if not data_filename or not run_identifier: return
    if not os.path.exists(data_filename): logging.error(f"Data file not found: {data_filename}"); return

    logging.info(f"Loading data from: {data_filename}")
    try:
        with open(data_filename, 'r') as f: data = json.load(f)
        logging.info(f"Loaded {len(data)} entries.")
    except Exception as e: logging.error(f"Failed to load data: {e}"); return
    if not data: logging.warning("Loaded data empty."); return

    all_coeffs, all_eigs = [], []
    logging.info("Extracting data...")
    for entry in data:
        all_coeffs.append(entry.get("coefficients") if isinstance(entry.get("coefficients"), list) else None)
        all_eigs.append(entry.get("eigenvalues") if isinstance(entry.get("eigenvalues"), list) else None)
    logging.info("Finished extracting data.")

    main_plot_dir = "plots"; run_plot_dir = os.path.join(main_plot_dir, run_identifier)
    try: os.makedirs(run_plot_dir, exist_ok=True); logging.info(f"Ensured plot directory exists: {run_plot_dir}")
    except OSError as e: logging.error(f"Could not create plot dir: {e}"); return

    # --- Generate Overall Coefficient Plot ---
    # (No changes here)
    coeffs_x, coeffs_y, coeffs_z, coeffs_color = [], [], [], []
    num_coeffs_dim = 0
    for coeffs in all_coeffs:
        if coeffs and len(coeffs) >= 3:
             num_coeffs_dim = max(num_coeffs_dim, len(coeffs))
             coeffs_x.append(coeffs[0]); coeffs_y.append(coeffs[1]); coeffs_z.append(coeffs[2])
             coeffs_color.append(coeffs[3] if len(coeffs) >= 4 else None)
    if len(coeffs_x) > 0:
        logging.info("Generating overall coefficient plot...")
        fig_coeffs = go.Figure()
        marker_config = dict(size=5)
        use_color = num_coeffs_dim >= 4 and any(c is not None for c in coeffs_color)
        if use_color: marker_config.update(dict(color=coeffs_color, colorscale='Viridis', showscale=True, colorbar_title=f"Coefficient {4}"))
        fig_coeffs.add_trace(go.Scatter3d(x=coeffs_x, y=coeffs_y, z=coeffs_z, mode='markers', marker=marker_config))
        fig_coeffs.update_layout(title=f'Coefficient Space ({run_identifier})', scene=dict(xaxis_title=f"Coefficient {1}", yaxis_title=f"Coefficient {2}", zaxis_title=f"Coefficient {3}"), margin=dict(l=0, r=0, b=0, t=40))
        plot_filename = os.path.join(run_plot_dir, "coeffs_overall.html")
        try: fig_coeffs.write_html(plot_filename, include_mathjax='cdn'); logging.info(f"Saved coefficient plot to: {plot_filename}")
        except Exception as e: logging.error(f"Failed to save coefficient plot: {e}")
    else: logging.warning("No sufficient coefficient data for overall plot.")

    # --- Generate Overall Eigenvalue Plot ---
    eigs_x, eigs_y, eigs_z, eigs_color = [], [], [], []
    num_eigs_dim = 0
    for eigs in all_eigs:
         if eigs and len(eigs) >= 3:
             num_eigs_dim = max(num_eigs_dim, len(eigs))
             eigs_x.append(eigs[0]); eigs_y.append(eigs[1]); eigs_z.append(eigs[2])
             eigs_color.append(eigs[3] if len(eigs) >= 4 else None)
    if len(eigs_x) > 0:
        logging.info("Generating overall eigenvalue plot...")
        fig_eigs = go.Figure()
        marker_config = dict(size=5)
        use_color = num_eigs_dim >= 4 and any(c is not None for c in eigs_color)
        eig_colorbar_title = None
        if use_color:
            # --- Revert to plain text ---
            eig_colorbar_title = "Eigenvalue 5"
            marker_config.update(dict(color=eigs_color, colorscale='Plasma', showscale=True, colorbar_title=eig_colorbar_title))
        fig_eigs.add_trace(go.Scatter3d(x=eigs_x, y=eigs_y, z=eigs_z, mode='markers', marker=marker_config))
        fig_eigs.update_layout(
            title=f'Eigenvalue Space ({run_identifier})',
            scene=dict(
                # --- Revert to plain text ---
                xaxis_title="Eigenvalue 2",
                yaxis_title="Eigenvalue 3",
                zaxis_title="Eigenvalue 4"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        plot_filename = os.path.join(run_plot_dir, "eigs_overall.html")
        try: fig_eigs.write_html(plot_filename, include_mathjax='cdn'); logging.info(f"Saved eigenvalue plot to: {plot_filename}")
        except Exception as e: logging.error(f"Failed to save eigenvalue plot: {e}")
    else: logging.warning("No sufficient eigenvalue data for overall plot.")

    # --- Generate Eigenvalue Slices ---
    if num_slices > 0 and num_eigs_dim >= 4:
        logging.info(f"Generating {num_slices} eigenvalue slice plots based on proximity (tolerance={slice_tolerance:.2e})...")
        valid_eigs_slice_dim = [val for val in eigs_x if isinstance(val, (int, float)) and not math.isnan(val)]
        if not valid_eigs_slice_dim: logging.warning("No valid Eigenvalue 2 data found for slicing.")
        else:
            min_eig_slice = min(valid_eigs_slice_dim); max_eig_slice = max(valid_eigs_slice_dim)
            logging.info(f"Slicing range for Eigenvalue 2: [{min_eig_slice:.4f}, {max_eig_slice:.4f}]")
            target_points = []
            if num_slices == 1: target_points.append((min_eig_slice + max_eig_slice) / 2.0)
            elif abs(max_eig_slice - min_eig_slice) < 1e-9: target_points.append(min_eig_slice); logging.warning("Range too small, using 1 slice.")
            else: step = (max_eig_slice - min_eig_slice) / (num_slices - 1); target_points = [min_eig_slice + i * step for i in range(num_slices)]

            for i, target_point in enumerate(target_points):
                slice_start_time = time.time()
                slice_l3, slice_l4, slice_l5 = [], [], []
                indices_in_slice = []
                for idx, eigs in enumerate(all_eigs):
                    if eigs and len(eigs) >= 4 and isinstance(eigs[0], (int, float)) and not math.isnan(eigs[0]):
                        if abs(eigs[0] - target_point) <= slice_tolerance:
                            if all(isinstance(e, (int, float)) and not math.isnan(e) for e in eigs[1:4]):
                                indices_in_slice.append(idx); slice_l3.append(eigs[1]); slice_l4.append(eigs[2]); slice_l5.append(eigs[3])

                if len(slice_l3) > 0:
                    logging.info(f"  Slice Point {i+1} (Eig2 ≈ {target_point:.4f}): Plotting {len(slice_l3)} points within tolerance {slice_tolerance:.2e}")
                    fig_slice = go.Figure()
                    marker_config_slice = dict(size=5, color='darkcyan', showscale=False)

                    # --- Update Hover Text to plain text ---
                    hover_texts = []
                    for idx in indices_in_slice:
                        if idx < len(all_eigs) and all_eigs[idx] and len(all_eigs[idx]) >= 4:
                             e = all_eigs[idx]
                             hover_texts.append(f"Eig2: {e[0]:.4f}<br>Eig3: {e[1]:.4f}<br>Eig4: {e[2]:.4f}<br>Eig5: {e[3]:.4f}")
                        else: hover_texts.append("Data unavailable")
                    # --- End Hover Text Update ---

                    fig_slice.add_trace(go.Scatter3d(x=slice_l3, y=slice_l4, z=slice_l5, mode='markers', marker=marker_config_slice, text=hover_texts, hoverinfo='text'))

                    # --- Update Titles/Labels to plain text ---
                    title = f"Eig Slice {i+1}/{len(target_points)} near Eig2={target_point:.4f} (Tol={slice_tolerance:.1e})"
                    plot_filename = os.path.join(run_plot_dir, f"eigs_slice_{i+1}.html")
                    fig_slice.update_layout(
                        title=title,
                        scene=dict(xaxis_title="Eigenvalue 3", yaxis_title="Eigenvalue 4", zaxis_title="Eigenvalue 5"),
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    # --- End Title/Label Update ---

                    try:
                        fig_slice.write_html(plot_filename, include_mathjax='cdn') # Keep MathJax in case other parts use it later? Or remove? Let's keep it for now.
                        logging.info(f"    Saved slice plot to: {plot_filename} (took {time.time()-slice_start_time:.2f}s)")
                    except Exception as e: logging.error(f"    Failed to save slice plot {plot_filename}: {e}")
                else: logging.info(f"  Slice Point {i+1} (Eig2 ≈ {target_point:.4f}): No valid points found within tolerance {slice_tolerance:.2e}")

    elif num_slices > 0 and num_eigs_dim < 4: logging.warning(f"Cannot generate slices: requires >= 4 eigenvalues, found {num_eigs_dim}.")
    elif num_slices == 0: logging.info("num_eigenvalue_slices is 0. Skipping slice generation.")

# --- Main Execution ---
if __name__ == "__main__":
    config_file_path = "config.toml"
    if not os.path.exists(config_file_path): print(f"Error: Config file '{config_file_path}' not found."); exit(1)
    create_plots(config_file_path)
    print("Plot generation finished.")