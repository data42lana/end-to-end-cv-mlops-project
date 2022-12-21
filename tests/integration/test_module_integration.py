import shutil
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml

# isort: off
from src.prepare_data import main as prepare_data
from src.optimize_hyperparams import main as optimize_hyperparams
from src.fine_tune_model import main as fine_tune_model
from src.model_test_inference import main as model_test_inference
from src.update_model_stages import main as update_model_stages


@pytest.fixture
def example_config():
    config_path = Path('tests/data_samples/example_config.yaml').absolute()
    with open(config_path) as conf:
        config = yaml.safe_load(conf)
    return config


@pytest.fixture
def data_config():
    conf = {'image_data_paths':
            {'info_csv_file': example_config['image_data_paths']['info_csv_file'],
             'bboxes_csv_file': example_config['image_data_paths']['bboxes_csv_file'],
             'train_csv_file': 'res/train.csv',
             'test_csv_file': 'res/val.csv'}}
    return conf


@pytest.mark.slow
def test_src_module_pipeline(example_config, data_config, train_df, val_df, tmp_path):
    # Arrange
    _ = shutil.copytree(Path.cwd().joinpath('tests/data_samples'), tmp_path / 'datas',
                        ignore=shutil.ignore_patterns('example_config.yaml'))

    mlflow.set_tracking_uri(f'sqlite:///{tmp_path}/tmlruns.db')
    exp_id = mlflow.create_experiment(example_config['mlflow_tracking_conf']['experiment_name'],
                                      tmp_path.as_uri())

    # Act
    prepare_data(tmp_path, data_config)
    optimize_hyperparams(tmp_path, example_config)
    fine_tune_model(tmp_path, example_config)
    model_test_inference(tmp_path, example_config, show_random_predict=True)
    update_model_stages(example_config)

    # Result
    prepared_train_df, prepared_test_df = [
        pd.read_csv(tmp_path.joinpath('res/' + f)) for f in ['train.csv', 'val.csv']]
    client = mlflow.MlflowClient()
    model_reg_info = client.get_latest_versions('best_tfrcnn')
    test_res_run = client.search_runs(
        [exp_id], "attributes.run_name='test-fine-tuning'")[0].info.run_id
    mlst_version = max([m.version for m in model_reg_info])

    assert prepared_train_df.equals(train_df) and prepared_test_df.equals(val_df)
    assert (tmp_path / 'res/hyper_opt_studies.db').exists()
    assert len([ch for ch in (tmp_path / 'res/tfrcnn_study/plots').iterdir()]) == 7
    assert (tmp_path / 'res/best_params.yaml').exists()
    assert [ch for ch in (tmp_path / 'res/val_outs').iterdir()]
    assert client.get_metric_history(test_res_run, 'f_beta')
    assert [ch for ch in (tmp_path / 'res/test_outs').iterdir()]
    assert client.get_model_version('best_tfrcnn', mlst_version).current_stage == 'Production'
