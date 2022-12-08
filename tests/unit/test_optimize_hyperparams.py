import pytest
import optuna
import yaml

from src.optimize_hyperparams import Objective, save_best_hyper_params, save_study_plots

def test_objective(dataloader, frcnn_model, hp_conf):
    objective = Objective(dataloader, dataloader, frcnn_model, hp_conf)
    assert 1 == objective(optuna.trial.FixedTrial({'optimizer': 'SGD', 'lr': 0.005, 'lr_scheduler': 'None'}))

def test_save_best_hyper_params_file_exists(simple_study, hp_conf, tmp_path):
    fpath = tmp_path / 'best_hp.yaml'
    save_best_hyper_params(simple_study, hp_conf, fpath)
    assert fpath.exists()

def test_save_best_hyper_params_file_content_structure(simple_study, hp_conf, tmp_path):
    bp_conf = yaml.load("""
        f_beta: 0.79
        optimizer:
            name: Adam
            parameters:
                lr: 0.001
        lr_scheduler: 
            name: 'None'    
    """)    
    fpath = tmp_path / 'best_hp.yaml'
    save_best_hyper_params(simple_study, hp_conf, fpath)
    with open(fpath) as f:
        content = yaml.safe_load(f)
    assert content == bp_conf

def test_save_study_plots(simple_study, tmp_path):
    fpath = tmp_path / 'tstudy'
    save_study_plots(simple_study, 'test_study', fpath)
    assert len([ch for ch in (fpath).iterdir()]) == 7
