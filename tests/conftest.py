from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope='session')
def bbox_path():
    fpath = Path('tests/data_samples/sample_bboxes.csv').absolute()
    return fpath


@pytest.fixture(scope='session')
def bbox_df(bbox_path):
    df = pd.read_csv(bbox_path)
    return df


@pytest.fixture(scope='session')
def train_csv_path():
    fpath = Path('tests/data_samples/sample_train.csv').absolute()
    return fpath


@pytest.fixture(scope='session')
def train_df(train_csv_path):
    df = pd.read_csv(train_csv_path)
    return df


@pytest.fixture(scope='session')
def val_df():
    fpath = Path('tests/data_samples/sample_val.csv').absolute()
    df = pd.read_csv(fpath)
    return df
