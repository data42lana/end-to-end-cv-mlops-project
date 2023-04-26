from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope='module')
def monitoring_data_path():
    fpath = Path('tests/data_samples/sample_monitoring_data.csv').absolute()
    return fpath


@pytest.fixture(scope='module')
def monitoring_data_df(monitoring_data_path):
    df = pd.read_csv(monitoring_data_path)
    return df
