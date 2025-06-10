import json
import os
import logging
import time
import math
import tomli

from lib import file_utils
from lib import plot_utils

if __name__ == "__main__":
    # 1. Load the config and setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    with open("config.toml", "rb") as f:
        config = tomli.load(f)

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

    # 3. Create the plots
    coef_plot = plot_utils.create_coeff_plot(n, all_coeffs, plot_dir)
    eig_plot = plot_utils.create_eig_plot(n, all_eigs, plot_dir)

    # 4. Save the plots
    try: 
        os.makedirs(plot_dir, exist_ok=True)
        logging.info(f"Ensured plot directory exists: {plot_dir}")
    except OSError as e: 
        logging.error(f"Could not create plot dir: {e}")
        exit()

    coef_plot.write_html(os.path.join(plot_dir, "coeffs.html"), include_mathjax='cdn')
    eig_plot.write_html(os.path.join(plot_dir, "eigs.html"), include_mathjax='cdn')

    print("Plot generation finished.")