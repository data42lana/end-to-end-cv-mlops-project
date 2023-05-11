"""This module updates raw data with new ones."""

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_param_config_yaml

logging.basicConfig(level=logging.INFO, filename='pipe.log',
                    format="%(asctime)s -- [%(levelname)s]: %(message)s")


def update_dir_or_csv_files(source, destination):
    """Update only a directory or CSV files, skipping all others."""
    if source.is_dir() and destination.is_dir():
        _ = shutil.copytree(source, destination, dirs_exist_ok=True)
    elif (source.suffix == '.csv') and (destination.suffix == '.csv'):
        updated_df = pd.concat([pd.read_csv(fpath) for fpath in [source, destination]])
        updated_df.to_csv(destination, index=False)


def main(project_path, param_config):
    """Update raw data with new ones."""
    # Get raw and new image data paths from the configurations
    img_data_paths = param_config['image_data_paths']
    new_img_data_paths = param_config['new_image_data_paths']

    # Check if new data exist
    new_data_exist = np.all([project_path.joinpath(new_img_data_paths[fpath]).exists()
                             for fpath in new_img_data_paths])

    # Update raw data
    if new_data_exist:

        for path in new_img_data_paths:
            data_path, new_data_path = [project_path / fpath[path]
                                        for fpath in [img_data_paths, new_img_data_paths]]
            update_dir_or_csv_files(new_data_path, data_path)

        logging.info("Raw data are updated.")


if __name__ == '__main__':
    project_path = Path.cwd()
    param_config = get_param_config_yaml(project_path)
    main(project_path, param_config)
