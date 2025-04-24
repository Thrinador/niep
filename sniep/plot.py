import json
import argparse
import plotly.graph_objects as go
import numpy as np
import logging
import os
import sys

# Attempt to import tomllib (Python 3.11+) or fall back to toml
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import toml
        # Define tomllib.load if using toml package
        class TomlDecodeError(Exception): pass # Define dummy exception
        class tomllib_wrapper:
             loads = toml.loads
             load = toml.load
             TOMLDecodeError = TomlDecodeError # Use dummy exception if toml doesn't expose its own easily
        tomllib = tomllib_wrapper()
        logging.info("Using 'toml' package for configuration loading.")
    except ModuleNotFoundError:
        logging.critical("Could not find 'tomllib' (Python 3.11+) or 'toml'. Please install 'toml' (`pip install toml`) if using Python < 3.11.")
        sys.exit(1)
else:
     logging.info("Using standard 'tomllib' for configuration loading.")


# --- Assume file_utils.py is in the same directory or Python path ---
# If not, you might need to adjust sys.path or copy the function
try:
    import file_utils
except ModuleNotFoundError:
    logging.critical("Could not import file_utils.py. Ensure it's in the same directory or your Python path.")
    # As a fallback, you could paste the build_file_name function here directly
    # def build_file_name(config, is_coef=True): ... (copy from file_utils.py)
    sys.exit(1)
# --- End file_utils import ---

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """Loads the TOML configuration file."""
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
    # Reuse the function from the previous version
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded data from {filepath}")
        if not isinstance(data, list):
            logging.warning("Loaded data is not a list. Assuming it's a single dictionary or invalid format.")
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

def process_eigenvalues(data, expected_n=4):
    """
    Extracts and validates eigenvalues for plotting.
    Expects data to be a list of dicts, each with an 'eigenvalues' key.
    For n=4, expects 3 eigenvalues in the list, already sorted.
    """
    # Reuse the function from the previous version
    lambda2_list = []
    lambda3_list = []
    lambda4_list = []
    expected_count = expected_n - 1
    processed_count = 0
    skipped_count = 0

    if not isinstance(data, list):
        logging.error("Invalid data format: Expected a list of dictionaries.")
        return None, None, None

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logging.warning(f"Skipping item #{i}: Not a dictionary.")
            skipped_count += 1
            continue

        eigenvalues = item.get('eigenvalues')

        if eigenvalues is None:
            logging.warning(f"Skipping item #{i}: Missing 'eigenvalues' key.")
            skipped_count += 1
            continue

        if not isinstance(eigenvalues, list):
            logging.warning(f"Skipping item #{i}: 'eigenvalues' is not a list.")
            skipped_count += 1
            continue

        if len(eigenvalues) != expected_count:
            logging.warning(f"Skipping item #{i}: Expected {expected_count} eigenvalues, found {len(eigenvalues)}.")
            skipped_count += 1
            continue

        try:
            lambda2, lambda3, lambda4 = map(float, eigenvalues)
        except (ValueError, TypeError):
             logging.warning(f"Skipping item #{i}: Eigenvalues could not be converted to float: {eigenvalues}")
             skipped_count += 1
             continue

        lambda2_list.append(lambda2)
        lambda3_list.append(lambda3)
        lambda4_list.append(lambda4)
        processed_count += 1

    logging.info(f"Processed {processed_count} eigenvalue sets, skipped {skipped_count}.")
    if not processed_count:
        logging.error("No valid eigenvalue sets found to plot.")
        return None, None, None

    return lambda2_list, lambda3_list, lambda4_list

def plot_eigenvalues(lambda2, lambda3, lambda4, output_filename="eigenvalue_plot_n4.html"):
    """Creates and saves a 3D scatter plot of eigenvalues."""
    # Reuse the function from the previous version
    if not lambda2 or not lambda3 or not lambda4:
        logging.error("No data provided for plotting.")
        return

    fig = go.Figure(data=[go.Scatter3d(
        x=lambda2,
        y=lambda3,
        z=lambda4,
        mode='markers',
        marker=dict(
            size=4,
            opacity=0.7,
        ),
        name='Eigenvalues'
    )])

    fig.update_layout(
        title='Eigenvalue Distribution for n=4 Stochastic Matrices ($\lambda_1=1$ removed)',
        scene=dict(
            xaxis_title='λ₂ (Largest non-unity)',
            yaxis_title='λ₃',
            zaxis_title='λ₄ (Smallest)',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    try:
        fig.write_html(output_filename)
        logging.info(f"Plot saved successfully to {output_filename}")
    except Exception as e:
        logging.error(f"Error saving plot to {output_filename}: {e}")

def main():
    """Main function to load config, build filename, load data, and plot."""
    parser = argparse.ArgumentParser(description="Plot n=4 eigenvalues, finding data file via config.toml.")
    # Changed argument to --config
    parser.add_argument("-c", "--config", default="config.toml", help="Path to the configuration TOML file (default: config.toml)")
    parser.add_argument("-o", "--output", default="eigenvalue_plot_n4.html", help="Output HTML file name (default: eigenvalue_plot_n4.html)")

    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args.config)
    if config is None:
        sys.exit(1)

    # 2. Check if n=4 in config
    try:
        n_value = config['global_data']['n']
        if n_value != 4:
            logging.error(f"Configuration file '{args.config}' specifies n={n_value}, but this script is designed for n=4.")
            sys.exit(1)
    except KeyError as e:
        logging.error(f"Could not find ['global_data']['n'] in configuration file '{args.config}': {e}")
        sys.exit(1)

    # 3. Build Eigenvalue Filename using file_utils
    eigenvalue_filename = file_utils.build_file_name(config, is_coef=False)
    if eigenvalue_filename is None:
        logging.error("Could not determine eigenvalue data filename from configuration.")
        sys.exit(1)
    logging.info(f"Expecting eigenvalue data in: {eigenvalue_filename}")

    # 4. Load Data using the generated filename
    data = load_eigenvalue_data(eigenvalue_filename)
    if data is None:
        sys.exit(1) # Exit if loading failed

    # 5. Process Eigenvalues
    l2, l3, l4 = process_eigenvalues(data, expected_n=4)
    if l2 is None:
        sys.exit(1) # Exit if processing failed

    # 6. Plot Eigenvalues
    plot_eigenvalues(l2, l3, l4, args.output)

if __name__ == "__main__":
    main()