"""This module updates raw data with new ones."""

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from utils import get_config_yml

logging.basicConfig(level=logging.INFO, filename='app.log',
                    format="[%(levelname)s]: %(message)s")


def update_dir_or_csv_files(source, destination):
    """Update only a directory or CSV files, skipping all others."""
    if source.is_dir() and destination.is_dir():
        _ = shutil.copytree(source, destination, dirs_exist_ok=True)
    elif (source.suffix == '.csv') and (destination.suffix == '.csv'):
        updated_df = pd.concat([pd.read_csv(fpath) for fpath in [source, destination]])
        updated_df.to_csv(destination, index=False)


def main(project_path, config):
    """Update raw data with new ones."""
    # Get raw and new image data paths from configurations
    img_data_paths = config['image_data_paths']
    new_img_data_paths = config['new_image_data_paths']

    # Check if new data exists
    new_data_exist = np.all([project_path.joinpath(new_img_data_paths[fpath]).exists()
                             for fpath in new_img_data_paths])

    # Update raw data
    if new_data_exist:

        for path in new_img_data_paths:
            data_path, new_data_path = [project_path / fpath[path]
                                        for fpath in [img_data_paths, new_img_data_paths]]
            update_dir_or_csv_files(new_data_path, data_path)

        logging.info("Raw data is updated.")


if __name__ == '__main__':
    project_path = Path.cwd()
    config = get_config_yml()
    main(project_path, config)
