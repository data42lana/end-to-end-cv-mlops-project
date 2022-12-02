import pytest
import pandas as pd
from pathlib import Path

# @pytest.fixture(scope='session')
# def imgs_path():
#     imgs_path = Path('tests/data_samples/sample_imgs').absolute()
#     return imgs_path

@pytest.fixture(scope='session')
def img_info_path():
    fpath = Path('tests/data_samples/sample_img_info.csv').absolute()
    return fpath

@pytest.fixture(scope='session')
def bbox_path():
    fpath = Path('tests/data_samples/sample_bboxes.csv').absolute()
    return fpath

@pytest.fixture(scope='session')
def train_csv_path():
    fpath = Path('tests/data_samples/sample_train.csv').absolute()
    return fpath

@pytest.fixture(scope='session')
def test_csv_path():
    fpath = Path('tests/data_samples/sample_test.csv').absolute()
    return fpath

@pytest.fixture(scope='session')
def img_info_df(img_info_path):
    df = pd.read_csv(img_info_path)
    return df

@pytest.fixture(scope='session')
def bbox_df(bbox_path):
    df = pd.read_csv(bbox_path)
    return df

@pytest.fixture(scope='session')
def train_df(train_csv_path):
    df = pd.read_csv(train_csv_path)
    return df

@pytest.fixture(scope='session')
def test_df(test_csv_path):
    df = pd.read_csv(test_csv_path)
    return df

@pytest.fixture()
def bbox_bbox_path(bbox_df):
    bbox_bbox_df = pd.concat([bbox_df, bbox_df], ignore_index=True)
    fpath = Path('bbox_bbox.csv')
    bbox_bbox_df.to_csv(fpath, index=False)
    yield fpath
    fpath.unlink()

# @pytest.fixture()
# def bbox_bbox_path(tmp_path, bbox_df):
#     bbox_bbox_df = pd.concat([bbox_df, bbox_df], ignore_index=True)
#     fpath = tmp_path / 'bbox_bbox.csv'
#     bbox_bbox_df.to_csv(fpath, index=False)
#     return fpath