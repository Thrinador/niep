import json
import logging
import os
import sys

def ensure_directory_exists(filename):
    """Creates the directory for the given filename if it doesn't exist."""
    parent_dir = os.path.dirname(filename)
    if parent_dir and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
            logging.info(f"Created directory: {parent_dir}")
        except OSError as e:
            logging.error(f"Could not create directory {parent_dir}: {e}")
            raise

def build_file_name(config, is_coef=True):
    """Builds the output filename based on configuration."""
    try:
        n = config['global_data']['n']
        points_dim = config['global_data']['points_dim']
        base = "ds-sniep"
        coefeig = "values" if is_coef else "eigenvalues"
        dims_str = "_".join(map(str, points_dim))
        base_dir = "data"

        filename = os.path.join(base_dir, f"{base}_{coefeig}_n{n}_dims{dims_str}.json")
        logging.debug(f"Generated filename: {filename} (is_coef={is_coef})")
        return filename
    except KeyError as e:
        logging.error(f"Missing expected key in config for building filename: {e}")
        return None
    except Exception as e:
        logging.exception("Error building filename:")
        return None

def save_results(results, filename):
    """Saves the results list (of dictionaries) to a JSON file."""
    if filename is None:
        logging.error("Cannot save results, filename is None.")
        return False
    logging.info(f"Preparing to save results to {filename}")
    try:
        ensure_directory_exists(filename)
        with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
        logging.info(f"Successfully saved results ({len(results)} items) to {filename}")
        return True
    except (IOError, OSError) as e:
        logging.exception(f"Failed to write results to {filename}:")
        return False
    except TypeError as e:
        logging.exception(f"TypeError during JSON serialization for {filename}. Check data types.")
        return False
    except Exception as e:
         logging.exception(f"An unexpected error occurred while saving results to {filename}:")
         return False

def load_results(filename):
    """Loads results from a JSON file."""
    if filename is None:
        logging.error("Cannot load results, filename is None.")
        return None
    logging.info(f"Attempting to load results from {filename}")
    try:
        with open(filename, 'r') as f:
            results_data = json.load(f)
        logging.info(f"Successfully loaded {len(results_data)} entries from {filename}.")
        return results_data
    except FileNotFoundError:
        logging.error(f"Results file not found: {filename}.")
        return None
    except (json.JSONDecodeError, TypeError) as e:
         logging.exception(f"Error reading or parsing JSON from {filename}:")
         return None
    except Exception as e:
        logging.exception(f"Unexpected error loading results file {filename}:")
        return None