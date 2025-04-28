# reformat_eigenvalue_files_cwd.py

import json
import os
import glob
import argparse
import re
import logging
import collections.abc
import time # For timing

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function for rounding INDIVIDUAL numbers/lists ---
def round_float_via_string(f_num, precision):
    """Converts float -> string -> float for cleaner representation."""
    if not isinstance(f_num, float): return f_num
    try:
        return float(f"{f_num:.{precision}f}")
    except (ValueError, TypeError):
        logging.warning(f"Could not format/convert float {f_num} to precision {precision}. Returning original.")
        return f_num

def round_float_list(flist, precision):
     """Applies rounding to a list of potential floats."""
     if not isinstance(flist, list): return flist
     # Filter out None before attempting to round
     return [round_float_via_string(item, precision) for item in flist if item is not None]

# --- Function for extracting coefficients (handles S-keys or existing list) ---
def extract_coefficients(entry):
    """
    Extracts S*_constraint and S*_optimized values into a sorted list,
    or returns existing 'coefficients' list. Returns None otherwise.
    """
    # Prioritize existing 'coefficients' key if present
    if 'coefficients' in entry and isinstance(entry['coefficients'], list):
        return entry['coefficients']

    # Otherwise, look for S*_constraint / S*_optimized keys
    coeffs = {}
    pattern = re.compile(r"^S(\d+)_(constraint|optimized)$")
    found_keys = False
    for key, value in entry.items():
        match = pattern.match(key)
        if match:
            s_num, key_type = int(match.group(1)), match.group(2)
            sort_key = (s_num, 1 if key_type == "optimized" else 0)
            coeffs[sort_key] = value; found_keys = True

    if not found_keys: return None # Neither format found
    return [coeffs[key] for key in sorted(coeffs.keys())]

# --- Function to process a single eigenvalue file ---
def process_eigenvalue_file(eigenvalue_filename, precision):
    """
    Loads an eigenvalue file, reformats/rounds its entries, and saves the result.
    """
    logging.info(f"Processing: {os.path.basename(eigenvalue_filename)}")
    start_time = time.time()
    input_dir = os.path.dirname(eigenvalue_filename); input_dir = input_dir or "."
    base_name = os.path.basename(eigenvalue_filename)

    # Construct output filename by removing '_eigenvalues'
    pattern = r"^(.*)_eigenvalues_(n\d+_dims[\d_]+)\.json$"
    match = re.match(pattern, base_name)
    if not match:
        logging.error(f"Could not parse filename pattern to generate output name: {base_name}")
        return
    prefix, suffix = match.group(1), match.group(2)
    output_filename = os.path.join(input_dir, f"{prefix}_{suffix}.json")

    # --- Load Data ---
    try:
        logging.info(f"  Loading {base_name}...")
        load_start = time.time()
        with open(eigenvalue_filename, 'r') as f: input_data = json.load(f)
        logging.info(f"  Loaded {base_name} in {time.time() - load_start:.2f}s ({len(input_data)} entries)")
    except Exception as e:
        logging.error(f"Failed to load file {eigenvalue_filename}: {e}"); return

    # --- Process Entries ---
    logging.info(f"  Reformatting entries...")
    reformat_start = time.time()
    reformatted_data = []
    skipped_count = 0
    processed_count = 0

    for entry in input_data:
        # No success check needed per user instruction

        # 1. Extract & Round Coefficients
        raw_coefficients = extract_coefficients(entry)
        if raw_coefficients is None:
            logging.warning(f"Could not extract coefficients. Skipping entry: {entry}")
            skipped_count += 1; continue
        coefficients = round_float_list(raw_coefficients, precision)
        if coefficients is None:
             logging.warning(f"Rounding failed for coefficients {raw_coefficients}. Skipping entry.")
             skipped_count += 1; continue

        # 2. Extract, Format & Round Eigenvalues
        raw_eigenvalues = entry.get("eigenvalues")
        formatted_eigenvalues = None
        if isinstance(raw_eigenvalues, list) and raw_eigenvalues:
            # Check for old format [[v, 0.0], ...]
            if isinstance(raw_eigenvalues[0], list):
                try:
                    flat_eigs = [item[0] for item in raw_eigenvalues if isinstance(item, list) and len(item)>0]
                    formatted_eigenvalues = round_float_list(flat_eigs, precision)
                except (TypeError, IndexError) as e:
                    logging.warning(f"Error converting old eigenvalue format: {e}. No eigenvalues for entry.")
                    formatted_eigenvalues = None
            else: # Assume flat list format
                formatted_eigenvalues = round_float_list(raw_eigenvalues, precision)

        # 3. Extract & Round Matrix
        matrix = round_float_list(entry.get('matrix'), precision)
        entry_type = entry.get('type', 'unknown') # Get type

        # 4. Construct final entry
        new_entry = {
            "type": entry_type,
            "coefficients": coefficients,
            "matrix": matrix,
        }
        if formatted_eigenvalues is not None:
             new_entry["eigenvalues"] = formatted_eigenvalues

        reformatted_data.append(new_entry)
        processed_count += 1

    logging.info(f"  Reformatted {processed_count} entries, skipped {skipped_count} entries in {time.time() - reformat_start:.2f}s.")

    if not reformatted_data:
        logging.warning(f"No valid data processed for {eigenvalue_filename}. Skipping save."); return

    # --- Save Results (Rounding already done inline) ---
    logging.info(f"  Saving {len(reformatted_data)} entries to {output_filename}...")
    save_start = time.time()
    try:
        with open(output_filename, 'w') as f: json.dump(reformatted_data, f, indent=4)
        logging.info(f"  Successfully saved in {time.time() - save_start:.2f}s")
    except Exception as e:
        logging.error(f"Failed to save output file {output_filename}: {e}")

    logging.info(f"Finished processing {base_name} in {time.time() - start_time:.2f}s total.")


# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="Reformat old ds-sniep *_eigenvalues_*.json files found in the current directory.")
    parser.add_argument("-p", "--precision", type=int, required=True,
                        help="Number of decimal places for rounding floats.")
    args = parser.parse_args()
    if args.precision < 0: print("Error: Precision must be non-negative."); return

    current_dir = "."
    logging.info(f"Searching for *_eigenvalues_*.json files in current directory: {os.path.abspath(current_dir)}")

    # Find only eigenvalue files
    eigenvalue_files = glob.glob(os.path.join(current_dir, "*_eigenvalues_*.json"))

    if not eigenvalue_files:
        print(f"No '*_eigenvalues_*.json' files found in the current directory.")
        return

    # Process each eigenvalue file found
    for eig_file in eigenvalue_files:
        process_eigenvalue_file(eig_file, args.precision)

    print("Reformatting complete.")

if __name__ == "__main__":
    main()