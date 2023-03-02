from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.fixture(scope='session')
def imgs_path():
    imgs_path = Path('tests/data_samples/sample_imgs').absolute()
    return imgs_path


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


@pytest.fixture
def config_yaml_file(tmp_path):
    fname = 'tconfig.yaml'
    fpath = tmp_path / fname
    fpath.parent.mkdir(exist_ok=True)
    config_dict = {'image_data_paths': {'images': 'datas/images'}}
    with open(fpath, 'w') as f:
        yaml.safe_dump(config_dict, f)
    return fname
