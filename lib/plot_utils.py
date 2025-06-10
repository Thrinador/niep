import logging
import math
import plotly.graph_objects as go

def create_coeff_plot(n, all_coeffs, run_identifier):
    logging.info("Generating coefficient plot...")
    fig_coeffs = go.Figure()
    marker_config = dict(size=5)

    if n == 3:
        coeffs_x, coeffs_y = [], []
        for coeffs in all_coeffs:
            coeffs_x.append(coeffs[0])
            coeffs_y.append(coeffs[1])

        fig_coeffs.add_trace(go.Scatter(x=coeffs_x, y=coeffs_y, mode='markers', marker=marker_config))
        fig_coeffs.update_layout(title=f'Coefficient Space ({run_identifier})', 
                                scene=dict(
                                    xaxis_title=f"Coefficient 1", 
                                    yaxis_title="Coefficient 2", 
                                ), 
                                margin=dict(l=0, r=0, b=0, t=40))

    elif n == 4:
        coeffs_x, coeffs_y, coeffs_z = [], [], []
        for coeffs in all_coeffs:
            coeffs_x.append(coeffs[0])
            coeffs_y.append(coeffs[1])
            coeffs_z.append(coeffs[2])

        fig_coeffs.add_trace(go.Scatter3d(x=coeffs_x, y=coeffs_y, z=coeffs_z, mode='markers', marker=marker_config))
        fig_coeffs.update_layout(title=f'Coefficient Space ({run_identifier})', 
                                scene=dict(
                                    xaxis_title=f"Coefficient 1", 
                                    yaxis_title="Coefficient 2", 
                                    zaxis_title="Coefficient 3"
                                ), 
                                margin=dict(l=0, r=0, b=0, t=40))
    elif n == 5:
        coeffs_x, coeffs_y, coeffs_z, coeffs_color = [], [], [], []
        for coeffs in all_coeffs:
            coeffs_x.append(coeffs[0])
            coeffs_y.append(coeffs[1])
            coeffs_z.append(coeffs[2])
            coeffs_color.append(coeffs[3])

        marker_config.update(dict(color=coeffs_color, colorscale='Viridis', showscale=True, colorbar_title="Coefficient 4"))
        fig_coeffs.add_trace(go.Scatter3d(x=coeffs_x, y=coeffs_y, z=coeffs_z, mode='markers', marker=marker_config))
        fig_coeffs.update_layout(title=f'Coefficient Space ({run_identifier})', 
                                scene=dict(
                                    xaxis_title=f"Coefficient 1", 
                                    yaxis_title="Coefficient 2", 
                                    zaxis_title="Coefficient 3"
                                ), 
                                margin=dict(l=0, r=0, b=0, t=40))

    return fig_coeffs

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