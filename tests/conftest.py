from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope='session')
def val_csv_path():
    fpath = Path('tests/data_samples/sample_val.csv').absolute()
    return fpath


@pytest.fixture(scope='session')
def val_df(val_csv_path):
    df = pd.read_csv(val_csv_path)
    return df


@pytest.fixture(scope='session')
def train_val_csv_path():
    fpath = Path('tests/data_samples/sample_train_val.csv').absolute()
    return fpath


@pytest.fixture(scope='session')
def train_val_df(train_val_csv_path):
    df = pd.read_csv(train_val_csv_path)
    return df
