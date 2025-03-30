# main.py
import logging
import sys
import time
import tomli
import os
from datetime import datetime
import json

# Import tasks from other modules
import optimize_tasks
import eigenvalue_tasks
import file_utils # May not be directly called but good practice

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


def load_config(config_path="config.toml"):
    """Loads the TOML configuration file."""
    try:
        with open(config_path, "rb") as f:
            config_data = tomli.load(f)
        print(f"Configuration loaded successfully from {config_path}") # Print before logging is ready
        return config_data
    except FileNotFoundError:
        print(f"FATAL ERROR: Configuration file not found: {config_path}", file=sys.stderr)
        return None
    except tomli.TOMLDecodeError as e:
         print(f"FATAL ERROR: Failed to parse TOML file {config_path}: {e}", file=sys.stderr)
         return None
    except Exception as e:
        print(f"FATAL ERROR: Unexpected error loading configuration: {e}", file=sys.stderr)
        return None


if __name__ == '__main__':
    print("Script execution started.") # Before logging
    main_start_time = time.perf_counter()

    # 1. Load Configuration
    config = load_config("config.toml")
    if config is None:
        sys.exit(1) # Exit if config fails

    # 2. Setup Logging and dump config to log
    if not setup_logging(config):
         sys.exit(1) # Exit if logging setup fails

    try:
        # Use json.dumps for pretty printing the config dictionary
        # Sort keys for consistent order in logs
        config_str = json.dumps(config, indent=4, sort_keys=True)
        logging.debug("--- Configuration Loaded ---")
        # Log the formatted string. The newline helps separate it in the log file.
        logging.debug(f"Full configuration details:\n{config_str}")
        logging.debug("--------------------------")
    except Exception as e:
        # Log an error if formatting/logging the config fails, but don't necessarily exit
        logging.error(f"Could not format or log configuration details: {e}")

    # 3. Run Optimization Tasks
    logging.info("===== Starting Optimization Phase =====")
    opt_status = optimize_tasks.run_optimization(config) # Returns 0 for success, 1 for failure

    # 4. Run Eigenvalue Tasks (only if optimization succeeded)
    eig_status = 1 # Default to failure
    if opt_status == 0:
        logging.info("===== Starting Eigenvalue Computation Phase =====")
        eig_status = eigenvalue_tasks.run_eigenvalue_computation(config) # Returns 0 for success, 1 for failure
    else:
        logging.warning("Skipping eigenvalue computation due to optimization failure or partial results.")

    # 5. Report Final Status
    main_end_time = time.perf_counter()
    total_time = main_end_time - main_start_time
    logging.info("===== Script Execution Finished =====")
    if opt_status == 0 and eig_status == 0:
        logging.info(f"All tasks completed successfully in {total_time:.4f} seconds.")
        sys.exit(0) # Success
    else:
        logging.error(f"Script finished with errors in {total_time:.4f} seconds.")
        logging.error(f"Optimization Status: {'Success' if opt_status == 0 else 'Failure/Partial'}")
        logging.error(f"Eigenvalue Status: {'Success' if eig_status == 0 else ('Failure' if opt_status == 0 else 'Skipped')}")
        sys.exit(1) # Failure