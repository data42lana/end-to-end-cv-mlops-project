from pathlib import Path

import pytest
import yaml
import pandas as pd
import mlflow

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

def test_src_module_pipeline(example_config, train_df, val_df, tmp_path):
    # Arrange
    test_project_path = Path.cwd() / 'tests'
    example_data_config = {'image_data_paths': 
                           {'images': 'data_samples/sample_imgs', 
                           'info_csv_file': 'data_samples/sample_img_info.csv',
                           'bboxes_csv_file': 'data_samples/sample_bboxes.csv',
                           'train_csv_file': 'tmp/train.csv',
                           'test_csv_file': 'tmp/val.csv'}}
    mlflow.set_tracking_uri(f'sqlite:///{tmp_path}/tmlruns.db')
    exp_id = mlflow.create_experiment(example_config['mlflow_tracking_conf']['experiment_name'], tmp_path.as_uri())

    # Act
    prepare_data(test_project_path, example_data_config)
    optimize_hyperparams(test_project_path, example_config)
    fine_tune_model(test_project_path, example_config)
    model_test_inference(test_project_path, example_config, show_random_predict=True)
    update_model_stages(test_project_path, example_config)

    # Result
    prepared_train_df, prepared_test_df = [pd.read(test_project_path.joinpath('tmp/' + f)) for f in ['train.csv', 'val.csv']]
    client = mlflow.MlflowClient()
    model_reg_info = client.get_latest_versions('best_tfrcnn')
    mlatest_version = max([m.version for m in model_reg_info])

    assert prepared_train_df.equals(train_df) and prepared_test_df.equals(val_df)
    assert (test_project_path / 'tmp/tfrcnn_study/hyper_opt_studies.db').exists()
    assert len([ch for ch in (test_project_path / 'tmp/tfrcnn_study/plots').iterdir()]) == 7
    assert (test_project_path / 'tmp/best_params.yaml').exists()
    assert [ch for ch in (test_project_path / 'tmp/val_outs').iterdir()]
    assert client.get_metric_history(client.search_runs([exp_id], "name='test-fine-tuning'")[0].info.run_id, 'f_beta')
    assert [ch for ch in (test_project_path / 'tmp/test_outs').iterdir()]
    assert client.get_model_version('best_tfrcnn', mlatest_version).current_stage == 'Production'
