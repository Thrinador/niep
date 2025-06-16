import logging
import math
import plotly.graph_objects as go
import os
import json


from lib import file_utils

def run_plotting(config):
    n = config['global_data']['n']

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

    create_plots(n, plot_dir, "coefficients", all_coeffs)
    create_plots(n, plot_dir, "eigenvalues", all_eigs)

    print("Plot generation finished.")

def create_plots_helper_2d(run_identifier, plot_type, all_points, i, j):
    logging.info(f"Generating {plot_type} 2d plot for indicies {i+1}, {j+1}")
    coeffs_x, coeffs_y = [], []
    for coeffs in all_points:
        coeffs_x.append(coeffs[i])
        coeffs_y.append(coeffs[j])

    fig_coeffs = go.Figure()
    marker_config = dict(size=5)

    fig_coeffs.add_trace(go.Scatter(x=coeffs_x, y=coeffs_y, mode='markers', marker=marker_config))
    fig_coeffs.update_layout(title=f'{plot_type} space ({run_identifier}) for E_{i+1} and E_{j+1}', 
                            scene=dict(
                                xaxis_title=f"{plot_type} {i+1}", 
                                yaxis_title=f"{plot_type} {j+1}", 
                            ))

    fig_coeffs.write_html(os.path.join(run_identifier, f"{plot_type}_2d_{i+1}_{j+1}.html"), include_mathjax='cdn')

def create_plots_helper_3d(run_identifier, plot_type, all_points, i, j, k):
    logging.info(f"Generating {plot_type} 3d plot for indicies {i+1}, {j+1}, {k+1}")
    coeffs_x, coeffs_y, coeffs_z = [], [], []
    for coeffs in all_points:
        coeffs_x.append(coeffs[i])
        coeffs_y.append(coeffs[j])
        coeffs_z.append(coeffs[k])

    fig_coeffs = go.Figure()
    marker_config = dict(size=5)

    fig_coeffs.add_trace(go.Scatter3d(x=coeffs_x, y=coeffs_y, z=coeffs_z, mode='markers', marker=marker_config))
    fig_coeffs.update_layout(title=f'{plot_type} Space ({run_identifier}) for E_{i+1}, E_{j+1}, and E_{k+1}', 
                            scene=dict(
                                xaxis_title=f"{plot_type} {i+1}", 
                                yaxis_title=f"{plot_type} {j+1}", 
                                zaxis_title=f"{plot_type} {k+1}"
                            ), 
                            margin=dict(l=0, r=0, b=0, t=40))

    fig_coeffs.write_html(os.path.join(run_identifier, f"{plot_type}_3d_{i+1}_{j+j}_{k+1}.html"), include_mathjax='cdn')

def create_plots_helper_4d(run_identifier, plot_type, all_points, i, j, k, l):
    logging.info(f"Generating {plot_type} 4d plot for indicies {i+1}, {j+1}, {k+1}, {l+1}")
    coeffs_x, coeffs_y, coeffs_z, coeffs_color = [], [], [], []
    for coeffs in all_points:
        coeffs_x.append(coeffs[i])
        coeffs_y.append(coeffs[j])
        coeffs_z.append(coeffs[k])
        coeffs_color.append(coeffs[l])

    fig_coeffs = go.Figure()
    marker_config = dict(size=5)

    marker_config.update(dict(color=coeffs_color, colorscale='Viridis', showscale=True, colorbar_title=f"{plot_type} {l+1}"))
    fig_coeffs.add_trace(go.Scatter3d(x=coeffs_x, y=coeffs_y, z=coeffs_z, mode='markers', marker=marker_config))
    fig_coeffs.update_layout(title=f'{plot_type} Space ({run_identifier}) for E_{i+1}, E_{j+1}, E_{k+1}, and E_{l+1}', 
                            scene=dict(
                                xaxis_title=f"{plot_type} {i+1}", 
                                yaxis_title=f"{plot_type} {j+1}", 
                                zaxis_title=f"{plot_type} {k+1}"
                            ), 
                            margin=dict(l=0, r=0, b=0, t=40))


def create_plots(n, run_identifier, plot_type, all_points):
    logging.info(f"Generating {plot_type} plot...")
    fig_coeffs = go.Figure()
    marker_config = dict(size=5)

    # Make all the 2d plots
    if n >= 3:
        for i in range(len(all_points[0])):
            for j in range(i+1, len(all_points[0])):
                create_plots_helper_2d(run_identifier, plot_type, all_points, i,j)

    # Make all the 3d plots
    if n >= 4:
        for i in range(len(all_points[0])):
            for j in range(i+1, len(all_points[0])):
                for k in range(j+1, len(all_points[0])):
                    create_plots_helper_3d(run_identifier, plot_type, all_points, i,j,k)

    # Make all the 4d plots
    if n >= 5:
        for i in range(len(all_points[0])):
            for j in range(i+1, len(all_points[0])):
                for k in range(j+1, len(all_points[0])):
                    for l in range(k+1, len(all_points[0])):
                        create_plots_helper_4d(run_identifier, plot_type, all_points, i,j,k,l)


def create_eig_plot(n, all_eigs, run_identifier):
    logging.info("Generating overall eigenvalue plot...")
    fig_eigs = go.Figure()
    marker_config = dict(size=5)

    if n == 3:
        eigs_x, eigs_y, = [], []
        for eigs in all_eigs:
            eigs_x.append(eigs[0])
            eigs_y.append(eigs[1])

        fig_eigs.add_trace(go.Scatter(x=eigs_x, y=eigs_y, mode='markers', marker=marker_config))
        fig_eigs.update_layout(
            title=f'Eigenvalue Space ({run_identifier})',
            scene=dict(
                xaxis_title="Eigenvalue 2",
                yaxis_title="Eigenvalue 3",
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    
    elif n == 4:
        eigs_x, eigs_y, eigs_z,  = [], [], []
        for eigs in all_eigs:
            eigs_x.append(eigs[0])
            eigs_y.append(eigs[1])
            eigs_z.append(eigs[2])

        fig_eigs.add_trace(go.Scatter3d(x=eigs_x, y=eigs_y, z=eigs_z, mode='markers', marker=marker_config))
        fig_eigs.update_layout(
            title=f'Eigenvalue Space ({run_identifier})',
            scene=dict(
                xaxis_title="Eigenvalue 2",
                yaxis_title="Eigenvalue 3",
                zaxis_title="Eigenvalue 4"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

    elif n == 5:
        eigs_x, eigs_y, eigs_z, eigs_color = [], [], [], []
        for eigs in all_eigs:
            eigs_x.append(eigs[0])
            eigs_y.append(eigs[1])
            eigs_z.append(eigs[2])
            eigs_color.append(eigs[3])

        eig_colorbar_title = "Eigenvalue 5"
        marker_config.update(dict(color=eigs_color, colorscale='Plasma', showscale=True, colorbar_title=eig_colorbar_title))
        fig_eigs.add_trace(go.Scatter3d(x=eigs_x, y=eigs_y, z=eigs_z, mode='markers', marker=marker_config))
        fig_eigs.update_layout(
            title=f'Eigenvalue Space ({run_identifier})',
            scene=dict(
                xaxis_title="Eigenvalue 2",
                yaxis_title="Eigenvalue 3",
                zaxis_title="Eigenvalue 4"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    return fig_eigs