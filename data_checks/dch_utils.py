"""This module contains helper functions for data validation."""

import argparse

import yaml


def get_data_path_config_yaml(project_path, data_path_config_file_path='configs/params.yaml'):
    """Get data path configurations from the params.yaml or another .yaml file."""
    config_path = project_path / data_path_config_file_path
    with open(config_path) as conf:
        data_path_config = yaml.safe_load(conf)
    return data_path_config


def get_data_type_arg_parser():
    """Return an argument parser object with a type of data."""
    parser = argparse.ArgumentParser(
        description='Specify one data type from "raw", "prepared", or "new" to be checked.',
        add_help=False)
    parser.add_argument('--check_data_type', type=str, choices=['raw', 'prepared', 'new'],
                        default='raw', help='check raw, prepared, or new data')
    return parser
