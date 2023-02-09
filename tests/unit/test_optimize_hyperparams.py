import optuna
import pytest
import yaml

from src.train.optimize_hyperparams import (Objective, save_best_hyper_params,
                                            save_study_plots)


@pytest.mark.slow
def test_objective(dataloader, frcnn_model, hp_conf):
    objective = Objective(dataloader, dataloader, frcnn_model, hp_conf)
    fixed_trial = optuna.trial.FixedTrial(
        {'optimizer': 'SGD', 'lr': 0.005, 'lr_scheduler': 'None'})
    objective_res = objective(fixed_trial)
    assert objective_res >= 0.0 and objective_res <= 1.0


class TestSaveBestHyperParams:

    def test_save_best_hyper_params_file_exists(self, simple_study, hp_conf, tmp_path):
        fpath = tmp_path / 'best_hp.yaml'
        save_best_hyper_params(simple_study, hp_conf, fpath)
        assert fpath.exists()

    def test_save_best_hyper_params_file_content_structure(self, simple_study,
                                                           hp_conf, tmp_path):
        bp_conf = yaml.safe_load("""
            f_beta: 0.79
            optimizer:
                name: Adam
                parameters:
                    lr: 0.001
            lr_scheduler:
                name: null
                parameters: null
        """)
        fpath = tmp_path / 'best_hp.yaml'
        save_best_hyper_params(simple_study, hp_conf, fpath)
        with open(fpath, 'r') as f:
            content = yaml.safe_load(f)
        assert content == bp_conf


def test_save_study_plots(simple_study, tmp_path):
    save_study_plots(simple_study, 'test_study', tmp_path)
    assert len([ch for ch in (tmp_path / 'test_study/plots').iterdir()]) == 7
