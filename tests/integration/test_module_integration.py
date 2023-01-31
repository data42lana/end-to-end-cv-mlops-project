import shutil
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml

# isort: off
from src.data.update_raw_data import main as update_raw_data
from src.data.prepare_data import main as prepare_data
from src.train.optimize_hyperparams import main as optimize_hyperparams
from src.train.fine_tune_model import main as fine_tune_model
from src.model.model_test_inference import main as model_test_inference
from src.model.update_model_stages import main as update_model_stages


@pytest.fixture
def example_config():
    config_path = Path('tests/data_samples/example_config.yaml').absolute()
    with open(config_path) as conf:
        config = yaml.safe_load(conf)
    return config


@pytest.mark.slow
def test_src_module_pipeline(example_config, val_df, train_val_df, tmp_path):
    # Arrange
    _ = shutil.copytree(Path.cwd().joinpath('tests/data_samples'), tmp_path / 'datas',
                        ignore=shutil.ignore_patterns('*.yaml', '*val.csv', '*train.csv'))
    mlflow.set_tracking_uri(f'sqlite:///{tmp_path}/tmlruns.db')
    exp_id = mlflow.create_experiment(example_config['mlflow_tracking_conf']['experiment_name'],
                                      tmp_path.as_uri())

    # Act
    prepare_data(tmp_path, example_config)
    update_raw_data(tmp_path, example_config)
    optimize_hyperparams(tmp_path, example_config)
    fine_tune_model(tmp_path, example_config)
    _ = model_test_inference(tmp_path, example_config, show_random_predict=True)
    _ = update_model_stages(tmp_path, example_config, save_metric_plots=True)

    # Result
    prepared_test_df = (
        pd.read_csv(tmp_path / example_config['image_data_paths']['test_csv_file']))
    updated_train_df = (
        pd.read_csv(tmp_path / example_config['image_data_paths']['train_csv_file'])
        .sort_values('Name', ignore_index=True))
    client = mlflow.MlflowClient()
    model_reg_info = client.get_latest_versions('best_tfrcnn')
    test_res_run = client.search_runs(
        [exp_id], "attributes.run_name='test-fine-tuning'")[0].info.run_id
    mlst_version = max([m.version for m in model_reg_info])

    assert prepared_test_df.equals(val_df)
    assert updated_train_df.equals(train_val_df.sort_values('Name', ignore_index=True))
    assert (tmp_path / 'res/hyper_opt_studies.db').exists()
    assert len([ch for ch in (tmp_path / 'res/tfrcnn_study/plots').iterdir()]) == 7
    assert (tmp_path / 'res/best_params.yaml').exists()
    assert [ch for ch in (tmp_path / 'res/val_outs').iterdir()]
    assert client.get_metric_history(test_res_run, 'f_beta')
    assert [ch for ch in (tmp_path / 'res/test_outs').iterdir()]
    assert client.get_model_version('best_tfrcnn', mlst_version).current_stage == 'Production'
    assert len([ch for ch in (tmp_path / 'plots').iterdir()]) == 2
