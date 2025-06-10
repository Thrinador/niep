# main.py
import logging
import sys
import time
import tomli
import os
from datetime import datetime
import json

# Import tasks from other modules
from lib import file_utils
from lib import optimize_tasks
from lib import eigenvalue_tasks

def setup_logging(config):
    """Configures logging based on config and timestamp."""
    try:
        # Generate dynamic log filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        n = config['global_data']['n']
        points_dim = config['global_data']['points_dim']
        dims_str = "-".join(map(str, points_dim))
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file_name = os.path.join(log_dir, f"{timestamp}_sniep_n-{n}_dims-{dims_str}.log")

        # Configure logging
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_date_format = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(log_format, datefmt=log_date_format)

        logger = logging.getLogger() # Get root logger
        logger.setLevel(logging.DEBUG) # Set lowest level

        # Clear existing handlers (important if re-running in interactive session)
        if logger.hasHandlers():
             logger.handlers.clear()

        # Console Handler (INFO+)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler (DEBUG+)
        fh = logging.FileHandler(log_file_name, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logging.info(f"Logging configured. Console: INFO+, File (DEBUG+): {log_file_name}")
        return True

    except KeyError as e:
         print(f"FATAL ERROR: Config missing key needed for logging setup: {e}", file=sys.stderr)
         return False
    except OSError as e:
         print(f"FATAL ERROR: Could not create log directory/file: {e}", file=sys.stderr)
         return False
    except Exception as e:
         print(f"FATAL ERROR: Unexpected error during logging setup: {e}", file=sys.stderr)
         return False


if __name__ == '__main__':
    print("Script execution started.")
    main_start_time = time.perf_counter()

    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    if config is None: sys.exit(1)
    if not setup_logging(config): sys.exit(1)

    try:
        config_str = json.dumps(config, indent=4, sort_keys=True)
        logging.debug(f"Full configuration details:\n{config_str}")
    except Exception as e:
        logging.error(f"Could not format or log configuration details: {e}")

    # --- Modified Workflow ---
    optimization_results = None
    final_results = None
    save_status = False

    # 1. Run Optimization Tasks
    logging.info("===== Starting Optimization Phase =====")
    optimization_results = optimize_tasks.run_optimization(config) # Returns list or None

    # 2. Run Eigenvalue Tasks (if optimization succeeded)
    if optimization_results is not None:
        logging.info("===== Starting Eigenvalue Computation Phase =====")
        # Pass optimization results list directly
        final_results = eigenvalue_tasks.run_eigenvalue_computation(config, optimization_results) # Returns updated list or None
    else:
        logging.warning("Skipping eigenvalue computation due to optimization failure or no results.")

    # 3. Save Final Results (if eigenvalue computation succeeded)
    if final_results is not None:
        logging.info("===== Saving Final Combined Results =====")
        output_filename = file_utils.build_file_name(config) # Get the single filename
        if output_filename:
            save_status = file_utils.save_results(config, final_results, output_filename)
            if not save_status:
                 logging.error(f"Failed to save final results to {output_filename}")
        else:
            logging.error("Failed to generate output filename. Cannot save results.")
            save_status = False # Explicitly mark as failure
    elif optimization_results is not None:
         # Optimization succeeded, but eigenvalues failed or produced nothing
         logging.error("Eigenvalue computation failed or produced no results. No final file saved.")
         save_status = False
    else:
        # Optimization failed
        logging.error("Optimization failed. No final file saved.")
        save_status = False

    # 4. Report Final Status
    main_end_time = time.perf_counter()
    total_time = main_end_time - main_start_time
    logging.info("===== Script Execution Finished =====")

    # Determine overall success based on final results being generated AND saved
    if final_results is not None and save_status:
        logging.info(f"All tasks completed and results saved successfully in {total_time:.4f} seconds.")
        sys.exit(0) # Success
    else:
        logging.error(f"Script finished with errors or incomplete results in {total_time:.4f} seconds.")
        if optimization_results is None:
             logging.error("Optimization Status: Failed")
        elif final_results is None:
             logging.error("Optimization Status: Succeeded")
             logging.error("Eigenvalue Status: Failed or No Results")
        elif not save_status:
             logging.error("Optimization Status: Succeeded")
             logging.error("Eigenvalue Status: Succeeded")
             logging.error("Save Status: Failed")
        sys.exit(1) # Failure