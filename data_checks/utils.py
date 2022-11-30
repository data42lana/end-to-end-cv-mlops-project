"""This module contains helper functions for data validation."""

import argparse
from pathlib import Path

import yaml

def get_config_yml(project_path):
    """Get configurations from a yaml file."""
    config_path = Path(project_path) / 'configs/config.yaml'
    with open(project_path / config_path) as conf:
        config = yaml.safe_load(conf)
    return config

def get_data_type_arg_parser():
    """Return a argument parser object with a type of data."""
    parser = argparse.ArgumentParser(
        description='Specify a type of data to check.',
        add_help=False)
    parser.add_argument('--check_data_type', type=str, choices=['raw', 'prepared', 'new'],
                        default='raw', help='check raw, prepared, or new data')
    return parser