import json
import argparse
import plotly.graph_objects as go
import numpy as np
import logging
import os
import sys

# (Keep tomllib/toml import logic as before)
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import toml
        class TomlDecodeError(Exception): pass
        class tomllib_wrapper:
             loads = toml.loads
             load = toml.load
             TOMLDecodeError = TomlDecodeError
        tomllib = tomllib_wrapper()
        logging.info("Using 'toml' package for configuration loading.")
    except ModuleNotFoundError:
        logging.critical("Could not find 'tomllib' (Python 3.11+) or 'toml'. Please install 'toml' (`pip install toml`) if using Python < 3.11.")
        sys.exit(1)
else:
     logging.info("Using standard 'tomllib' for configuration loading.")

# (Keep file_utils import logic as before)
try:
    import file_utils
except ModuleNotFoundError:
    logging.critical("Could not import file_utils.py. Ensure it's in the same directory or your Python path.")
    sys.exit(1)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """Loads the TOML configuration file."""
    # Reuse function
    try:
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
        logging.info(f"Configuration loaded successfully from {config_path}")
        return config_data
    except FileNotFoundError:
        logging.error(f"FATAL ERROR: Configuration file not found: {config_path}")
        return None
    except tomllib.TOMLDecodeError as e:
         logging.error(f"FATAL ERROR: Failed to parse TOML file {config_path}: {e}")
         return None
    except Exception as e:
        logging.error(f"FATAL ERROR: Unexpected error loading configuration {config_path}: {e}")
        return None

def load_eigenvalue_data(filepath):
    """Loads eigenvalue data from a JSON file."""
    # Reuse function
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded data from {filepath}")
        if not isinstance(data, list):
            logging.warning("Loaded data is not a list.")
            return None
        return data
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {filepath}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred loading {filepath}: {e}")
        return None

# Modified to extract S_k values instead of calculating coefficients
def process_data_n5(data, expected_n=5, eig_sum_threshold=-0.1):
    """
    Extracts eigenvalues and S_k values, filters by eigenvalue sum.
    For n=5, expects 4 eigenvalues [l2, l3, l4, l5] and keys
    'S1_constraint', 'S2_constraint', 'S3_constraint', 'S4_optimized'.
    Filters out entries where l2+l3+l4+l5 < eig_sum_threshold.
    """
    l2_list, l3_list, l4_list, l5_list = [], [], [], []
    # Lists for S_k values
    s1_list, s2_list, s3_list, s4_list = [], [], [], []

    expected_eig_count = expected_n - 1 # Should be 4 for n=5
    # Define the keys we expect for the S values
    s_keys = ['S1_constraint', 'S2_constraint', 'S3_constraint', 'S4_optimized']

    processed_count = 0
    skipped_count = 0
    filtered_count = 0

    if not isinstance(data, list):
        logging.error("Invalid data format: Expected a list of dictionaries.")
        return None # Indicate failure

    for i, item in enumerate(data):
        valid_entry = True
        log_skip_reason = ""

        if not isinstance(item, dict):
            log_skip_reason = f"Item #{i}: Not a dictionary."
            valid_entry = False
        else:
            eigenvalues = item.get('eigenvalues')
            if eigenvalues is None or not isinstance(eigenvalues, list) or len(eigenvalues) != expected_eig_count:
                log_skip_reason = f"Item #{i}: Invalid/missing 'eigenvalues' (expected list of {expected_eig_count}). Found: {eigenvalues}"
                valid_entry = False

            # Check for S_k keys
            s_values = {}
            for key in s_keys:
                if key not in item:
                    log_skip_reason = f"Item #{i}: Missing expected key '{key}'."
                    valid_entry = False
                    break
                s_values[key] = item[key]

        if not valid_entry:
            logging.warning(f"Skipping {log_skip_reason}")
            skipped_count += 1
            continue

        try:
            # Extract eigenvalues
            l2, l3, l4, l5 = map(float, eigenvalues)

            # *** Eigenvalue Sum Check ***
            eig_sum = 1+ l2 + l3 + l4 + l5
            if eig_sum < eig_sum_threshold:
                logging.debug(f"Filtering item #{i}: Eigenvalue sum ({eig_sum:.4f}) < threshold ({eig_sum_threshold}).")
                filtered_count += 1
                continue # Skip this entry

            # Extract S_k values
            s1 = float(s_values['S1_constraint'])
            s2 = float(s_values['S2_constraint'])
            s3 = float(s_values['S3_constraint'])
            s4 = float(s_values['S4_optimized'])

        except (ValueError, TypeError) as e:
             logging.warning(f"Skipping item #{i}: Eigenvalues or S_k values could not be converted to float: {e}")
             skipped_count += 1
             continue

        # Append to lists only if not filtered
        l2_list.append(l2)
        l3_list.append(l3)
        l4_list.append(l4)
        l5_list.append(l5)
        s1_list.append(s1)
        s2_list.append(s2)
        s3_list.append(s3)
        s4_list.append(s4)
        processed_count += 1

    logging.info(f"Processed {processed_count} entries, skipped {skipped_count} due to format/errors, filtered {filtered_count} by eigenvalue sum.")
    if not processed_count:
        logging.error("No valid entries remaining after processing and filtering.")
        return None # Indicate failure

    # Return all lists needed for plotting
    return l2_list, l3_list, l4_list, l5_list, s1_list, s2_list, s3_list, s4_list

# Eigenvalue plot function remains the same as the previous version
def plot_eigenvalues_n5(l2, l3, l4, l5, output_filename="eigenvalue_plot_n5.html"):
    """Creates and saves a 3D scatter plot of eigenvalues (n=5) with coordinate hover text."""
    if not all(lst for lst in [l2, l3, l4, l5]):
        logging.error("Missing data for eigenvalue plot.")
        return

    hover_template_eig = (
        '<b>λ₂</b>: %{x:.4f}<br>'
        '<b>λ₃</b>: %{y:.4f}<br>'
        '<b>λ₄</b>: %{z:.4f}<br>'
        '<b>λ₅</b>: %{marker.color:.4f}'
        '<extra></extra>'
    )

    fig = go.Figure(data=[go.Scatter3d(
        x=l2, y=l3, z=l4,
        mode='markers',
        marker=dict(
            size=4,
            color=l5,
            colorscale='Viridis',
            colorbar_title='λ₅',
            showscale=True,
            opacity=0.7
        ),
        hovertemplate=hover_template_eig
    )])

    fig.update_layout(
        title='Eigenvalue Distribution n=5, λ₁=1 removed',
        scene=dict(
            xaxis_title='λ₂',
            yaxis_title='λ₃',
            zaxis_title='λ₄',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    try:
        fig.write_html(output_filename)
        logging.info(f"Eigenvalue plot saved successfully to {output_filename}")
    except Exception as e:
        logging.error(f"Error saving eigenvalue plot to {output_filename}: {e}")

# Renamed and modified function for plotting S_k values
def plot_S_values_n5(s1, s2, s3, s4, output_filename="s_values_plot_n5.html"):
    """Creates and saves a 3D scatter plot of S_k values (n=5) with coordinate hover text."""
    if not all(lst for lst in [s1, s2, s3, s4]):
        logging.error("Missing data for S-value plot.")
        return

    # Define hover template for S values
    hover_template_s = (
        '<b>S₁ Constraint</b>: %{x:.4f}<br>'
        '<b>S₂ Constraint</b>: %{y:.4f}<br>'
        '<b>S₃ Constraint</b>: %{z:.4f}<br>'
        '<b>S₄ Optimized</b>: %{marker.color:.4f}'
        '<extra></extra>'
    )

    fig = go.Figure(data=[go.Scatter3d(
        x=s1, y=s2, z=s3,
        mode='markers',
        marker=dict(
            size=4,
            color=s4,                # Set color to S4 values
            colorscale='Plasma',     # Keep different colorscale
            colorbar_title='S₄ Optimized', # Updated label
            showscale=True,
            opacity=0.7
        ),
        hovertemplate=hover_template_s # Use the S-value template
    )])

    fig.update_layout(
        title='S-Value Distribution n=5, from Optimization',
        scene=dict(
            xaxis_title='S₁ Constraint', # Updated axis label
            yaxis_title='S₂ Constraint', # Updated axis label
            zaxis_title='S₃ Constraint', # Updated axis label
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    try:
        fig.write_html(output_filename)
        logging.info(f"S-value plot saved successfully to {output_filename}")
    except Exception as e:
        logging.error(f"Error saving S-value plot to {output_filename}: {e}")

# Modified main function
def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Plot n=5 eigenvalues and S-values, finding data file via config.toml.")
    parser.add_argument("-c", "--config", default="config.toml", help="Path to the configuration TOML file (default: config.toml)")
    parser.add_argument("--eig-out", default="eigenvalue_plot_n5.html", help="Output HTML file name for eigenvalue plot")
    # Updated argument name for clarity
    parser.add_argument("--sval-out", default="s_values_plot_n5.html", help="Output HTML file name for S-values plot")

    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args.config)
    if config is None: sys.exit(1)

    # 2. Check if n=5 in config
    try:
        n_value = config['global_data']['n']
        if n_value != 5:
            logging.error(f"Configuration file '{args.config}' specifies n={n_value}, but this script is designed for n=5.")
            sys.exit(1)
    except KeyError as e:
        logging.error(f"Could not find ['global_data']['n'] in configuration file '{args.config}': {e}")
        sys.exit(1)

    # 3. Build Eigenvalue Filename
    eigenvalue_filename = file_utils.build_file_name(config, is_coef=False)
    if eigenvalue_filename is None:
        logging.error("Could not determine eigenvalue data filename from configuration.")
        sys.exit(1)
    logging.info(f"Expecting eigenvalue data in: {eigenvalue_filename}")

    # 4. Load Data
    data = load_eigenvalue_data(eigenvalue_filename)
    if data is None: sys.exit(1)

    # 5. Process Data (Filter, Eigenvalues, S-Values)
    processed_data = process_data_n5(data, expected_n=5, eig_sum_threshold=-0.1)
    if processed_data is None: sys.exit(1)
    # Unpack results, including S-values now
    l2, l3, l4, l5, s1, s2, s3, s4 = processed_data

    # 6. Plot Eigenvalues
    plot_eigenvalues_n5(l2, l3, l4, l5, args.eig_out)

    # 7. Plot S-Values (calling the renamed function)
    plot_S_values_n5(s1, s2, s3, s4, args.sval_out)

if __name__ == "__main__":
    main()