"""This module contains helper functions for model monitoring."""

import pandas as pd
import yaml


def get_monitoring_param_config_yaml(project_path,
                                     monitoring_config_file_path='configs/params.yaml'):
    """Get monitoring configurations from the params.yaml or another .yaml file."""
    config_path = project_path / monitoring_config_file_path
    with open(config_path) as conf:
        monitoring_param_config = yaml.safe_load(conf)
    return monitoring_param_config


def get_number_of_csv_rows(csv_file_path, read_column=None):
    """Return the number of rows in a CSV file."""
    if read_column is not None:
        read_column = [read_column]
    df = pd.read_csv(csv_file_path, usecols=read_column)
    return df.shape[0]
