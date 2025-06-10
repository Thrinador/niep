import json
import logging
import os
import sys
import collections.abc

def round_nested_floats(data, precision):
    """
    Recursively rounds floats to a specific precision by converting
    via a formatted string, returning a float. Handles nested structures.
    Other types (int, str, bool, None) are left unchanged.
    """
    # Check if the data is specifically a float
    if isinstance(data, float):
        try:
            # 1. Format the float to a string with the specified precision
            formatted_string = f"{data:.{precision}f}"
            # 2. Convert the formatted string back to a float
            cleaned_float = float(formatted_string)
            return cleaned_float
        except (ValueError, TypeError):
            # Handle potential errors during formatting or conversion
            logging.warning(f"Could not format/convert float {data} to precision {precision}. Returning original float.")
            return data # Return the original float if processing fails
    # If it's a dictionary-like object, recurse into its values
    elif isinstance(data, collections.abc.Mapping):
        return {k: round_nested_floats(v, precision) for k, v in data.items()}
    # If it's a list/tuple-like object (but not string/bytes), recurse into its items
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return [round_nested_floats(item, precision) for item in data]
    # Otherwise (int, str, bool, None, etc.), return the data unchanged
    else:
        return data

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

def build_file_name(config, file_type='data'):
    """Builds the consolidated output filename based on configuration."""
    return build_file_name_no_extension(config, file_type) + ".json"

def build_file_name_no_extension(config, file_type='data'):
    """Builds the consolidated output filename based on configuration."""
    try:
        n = config['global_data']['n']
        points_dim = config['global_data']['points_dim']
        base = config['global_data']['matrix_type']
        dims_str = "_".join(map(str, points_dim))
        base_dir = os.path.join(base,file_type)

        filename = os.path.join(base_dir, f"{base}_n{n}_dims{dims_str}")
        logging.debug(f"Generated consolidated filename: {filename}")
        return filename
    except KeyError as e:
        logging.error(f"Missing expected key in config for building filename: {e}")
        return None
    except Exception as e:
        logging.exception("Error building filename:")
        return None

def save_results(config, results, filename):
    """
    Saves the results list (of dictionaries) to a JSON file,
    applying rounding based on config['decimal_precision'].
    """
    if filename is None:
        logging.error("Cannot save results, filename is None.")
        return False
    if config is None:
        logging.error("Cannot save results, config is None.")
        return False

    logging.info(f"Preparing to save results to {filename}")

    precision = config['file_utils_data']['decimal_precision']
    if precision is not None:
        logging.debug(f"Rounding numerical data to {precision} decimal places before saving.")
        try:
            results_to_save = round_nested_floats(results, precision)
        except Exception as round_err:
             logging.error(f"Error during rounding: {round_err}. Saving unrounded data.")
             results_to_save = results # Fallback to unrounded data
    else:
        logging.warning("Config 'decimal_precision' not found. Saving unrounded data.")
        results_to_save = results

    try:
        ensure_directory_exists(filename)
        with open(filename, 'w') as f:
                json.dump(results_to_save, f, indent=4) # Save the potentially rounded data
        logging.info(f"Successfully saved results ({len(results_to_save)} items) to {filename}")
        return True
    except (IOError, OSError) as e:
        logging.exception(f"Failed to write results to {filename}:")
        return False
    except TypeError as e:
        # This might catch issues if rounding failed and left incompatible types
        logging.exception(f"TypeError during JSON serialization for {filename}. Check data types (potentially after rounding).")
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