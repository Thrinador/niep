import logging
import time
import tomli

from lib import file_utils
from lib import plot_utils


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    if config["plot_data"]["data_location"] is not "":
        plot_utils.run_plotting(config, config["plot_data"]["data_location"])
    else:
        plot_utils.run_plotting(config)
    