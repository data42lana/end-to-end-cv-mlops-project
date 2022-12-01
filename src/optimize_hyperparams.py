"""This module implements automatic hyperparameter optimization with Optuna."""

import sys
import random
from pathlib import Path
import logging

import yaml
import numpy as np
import optuna # Hyperparameter Optimization
import mlflow # Experiment Tracking
import torch # PyTorch

from image_dataloader import get_train_val_test_dataloaders
from object_detection_model import faster_rcnn_mob_model_for_n_classes
from train_inference_fns import train_one_epoch, eval_one_epoch
from utils import get_device, get_config_yml

# Set partial reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

PROJECT_PATH = Path.cwd()
# Get configurations
config = get_config_yml()

DEVICE = get_device(config['model_training_inference_conf']['device_cuda'])
BATCH_SIZE = config['image_dataset_conf']['batch_size']
NUM_CLASSES = config['object_detection_model']['number_classes']
EVAL_IOU_THRESH = config['model_training_inference_conf']['evaluation_iou_threshold']
EVAL_BETA = config['model_training_inference_conf']['evaluation_beta']

HYPER_OPT_PARAMS = config['hyperparameter_optimization']
HYPERPARAMS = HYPER_OPT_PARAMS['hyperparameters']
METRIC = HYPER_OPT_PARAMS['metric']
SAMPLER = HYPER_OPT_PARAMS['sampler']
PRUNER = HYPER_OPT_PARAMS['pruner']

# Log parameters and metrics during hyperparameter optimization into MLflow
mlflow.set_tracking_uri(PROJECT_PATH.joinpath(config['mlops_tracking_conf']['tracking_dir']).as_uri())
hyp_opt_exp = mlflow.get_experiment_by_name('Hyperparameter_Optimization')

if hyp_opt_exp is not None:
    hyp_opt_exp_id = hyp_opt_exp.experiment_id
else:
    hyp_opt_exp_id = mlflow.create_experiment('Hyperparameter_Optimization')

mlc = optuna.integration.mlflow.MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(), 
    metric_name=METRIC, create_experiment=False, 
    mlflow_kwargs={
        'experiment_id': hyp_opt_exp_id, 
        'run_name': '_'.join(['HO', SAMPLER['name'][:-7], PRUNER['name'][-6]]),
        'tags': {'model_name': config['object_detection_model']['name'],
                 'tools.training': 'PyTorch',
                 'sampler': SAMPLER['name'],
                 'pruner': PRUNER['name'],
                 'tools.hyper_opt': 'Optuna'}})

@mlc.track_in_mlflow()
def objective(trial):
    """The function to be optimized."""
    # Get dataloaders
    train_dl, val_dl, _ = get_train_val_test_dataloaders(BATCH_SIZE)

    model_params = config['object_detection_model']['load_parameters'] 
    frcnn_mob_model = faster_rcnn_mob_model_for_n_classes(NUM_CLASSES, **model_params)
    frcnn_mob_model.to(DEVICE)    
  
    trials_suggest = {'cat': trial.suggest_categorical,
                      'int': trial.suggest_int,
                      'float': trial.suggest_float}

    # Construct a training optimizer and a lr_scheduler
    optimizer_name = trial.suggest_categorical('optimizer', HYPERPARAMS['optimizers'].keys())
    optim_params = {k: trials_suggest[p[1]](k, **p[0]) for k, p in HYPERPARAMS['optimizers'][optimizer_name]}    

    lr_scheduler_name = trial.suggest_categorical('lr_scheduler', HYPERPARAMS['lr_schedulers'].keys())
    lr_scheduler_params = {k: trials_suggest[p[1]](k, **p[0]) for k, p in HYPERPARAMS['lr_scheduler'][lr_scheduler_name]}

    train_model_params = [p for p in frcnn_mob_model.parameters() if p.requires_grad]
    optimizer = getattr(torch.optim, optimizer_name)(train_model_params, **optim_params)

    if lr_scheduler_name is not None:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler_name)(optimizer, **lr_scheduler_params)
    else:
        lr_scheduler = None
    
    # Log parameters into MLflow
    mlflow.log_params({'seed': SEED,
                       'device': DEVICE,
                       'num_classes': NUM_CLASSES,
                       'batch_size': BATCH_SIZE})                     
    mlflow.log_params({'eval_iou_thresh': EVAL_IOU_THRESH,
                       'eval_beta': EVAL_BETA,
                       **model_params})
    mlflow.log_params({'optimizer': optimizer_name,
                       'lr_scheduler': lr_scheduler_name,
                       **lr_scheduler_params,
                       **optim_params})

    # Train the model
    for epoch in range(1, HYPER_OPT_PARAMS['epochs']+1):
        train_res = train_one_epoch(train_dl, frcnn_mob_model, optimizer, DEVICE)

        if lr_scheduler is not None:
            lr_scheduler.step()        

        eval_res = eval_one_epoch(val_dl, frcnn_mob_model, EVAL_IOU_THRESH , EVAL_BETA, DEVICE)
        opt_score = eval_res['epoch_scores'][METRIC]
        trial.report(opt_score, epoch)
        
        # Log metrics into MLflow
        mlflow.log_metric('train_epoch_loss', train_res['epoch_loss'], step=epoch)
        mlflow.log_metrics(train_res['epoch_dict_losses'], step=epoch)
        mlflow.log_metrics(eval_res['epoch_scores'], step=epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

    return opt_score

def save_study_plots(study, study_name, save_path):
    """Save study result plots."""
    plots = [optuna.visualization.plot_optimization_history,
             optuna.visualization.plot_intermediate_values,
             optuna.visualization.plot_parallel_coordinate,
             optuna.visualization.plot_contour,
             optuna.visualization.plot_slice,
             optuna.visualization.plot_param_importances,
             optuna.visualization.plot_edf]

    for plot in plots:
        fig = plot(study)
        fname = plot.__name__[5:]
        save_path = Path(save_path) / 'plots' / study_name
        save_path.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_path / f'{fname}.jpeg')

def main():
    """Run an optimization study."""
    hyper_opt_path = PROJECT_PATH / HYPER_OPT_PARAMS['save_study_dir']
    hyper_opt_path.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename='logs/hparam_opt_log.txt',
                        format="[%(levelname)s]: %(message)s")

    # Set study parameters
    study_callbacks=[mlc]
    if str(DEVICE) == 'cuda':
        study_callbacks.append(lambda study, trial: torch.cuda.empty_cache())
    study_storage = optuna.storages.RDBStorage(url='sqlite:///{}'.format(hyper_opt_path / 'hyper_opt_studies.db'))
    
    sampler_pruner = []
    for osp, sp in zip((optuna.samplers, optuna.pruners), (SAMPLER, PRUNER)):
        sp_params = sp['parameters'] if sp['parameters'] else {}
        sampler_pruner.append(getattr(osp, sp['name'])(**sp_params))

    # Run a optimization session
    study = optuna.create_study(direction='maximize', sampler=sampler_pruner[0],
                                pruner=sampler_pruner[1], storage=study_storage, 
                                study_name=HYPER_OPT_PARAMS['study_name'], 
                                load_if_exists=True)

    study.optimize(objective, n_trials=HYPER_OPT_PARAMS['n_trials'], 
                   timeout=HYPER_OPT_PARAMS['timeout'], callbacks=study_callbacks,
                   gc_after_trial=True)

    # Save the best parameters
    save_best_params_path = PROJECT_PATH / HYPER_OPT_PARAMS['save_best_parameters_path']
    save_best_params_path.mkdir(exist_ok=True)
    
    best_params = {METRIC: round(study.best_values, 2)}
    for hp in ['optimizer', 'lr_scheduler']:
        hps = {}
        for k in HYPERPARAMS[study.best_params[hp]].keys():
            hps[k]: study.best_params[k]            
        best_params[hp] = {study.best_params[hp]: hps}

    with open(save_best_params_path, 'w') as f:
        yaml.safe_dump(best_params, f)
    logging.info("The best parameters are saved.")
    
    # Save study visualizations
    save_study_plots(study, HYPER_OPT_PARAMS['study_name'], hyper_opt_path)
    logging.info("Plots are saved.")

if __name__ == '__main__':
    main()