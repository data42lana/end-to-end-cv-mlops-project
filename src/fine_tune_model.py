"""This module implements fine-tuning of object detection model."""

import gc
import random
from pathlib import Path

import yaml
import numpy as np
import mlflow # Experiment Tracking and Model Registry
import torch # PyTorch
import torchvision

from train_inference_fns import train_one_epoch, eval_one_epoch
from object_detection_model import faster_rcnn_mob_model_for_n_classes
from image_dataloader import get_train_val_test_dataloaders
from utils import save_model_state, draw_bboxes_on_image, get_device

CONFIG_PATH = 'configs/config.yaml'

# Set partial reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def run_train(train_dataloader, val_dataloader, model, epochs, optimizer_name, optim_params,
              save_best_model_path, lr_scheduler_name=None, lr_scheduler_params=None, 
              device=torch.device('cpu'), metric_to_find_best_model=None, 
              init_metric_value=0.0, eval_iou_thresh=0.5, eval_beta=1, model_name='best_model', 
              save_best_ckpt=False, checkpoint=None, log_metrics=False, register_best_log_model=False, 
              reg_model_name='best_model', show_random_best_model_prediction=False):
    """Runs a new training and evaluation cycle of a model for a fixed number of epochs
    or continue if checkpoint is passed, while saving the best model (or checkpoint).
    
    Parameters:
        train_dataloader (Dataloader): images, labels and boxes for a training step
        val_dataloader (Dataloader): images, labels and boxes for an evaluation step
        model (nn.Module): an object detection model
        epochs (int): number of training epochs
        optimizer_name (str): an optimizer name from torch.optim
        optim_params (dict): relevant parameters for the optimizer
        save_best_model_path (Path): a path to directory to save the best model or its checkpoint
        lr_scheduler_name (str) (optional): a learning rate scheduler name 
            from torch.optim.lr_scheduler (default None)
        lr_scheduler_params (dict) (optional): relevant parameters for 
            the learning rate scheduler (default None)
        device (torch.device): a type of device used: torch.device('cpu' or 'cuda') 
            (default torch.device('cpu'))
        metric_to_find_best_model (str) (optional): a corresponding model score is tracked 
            to find the best model (default None) 
        init_metric_value (float): an initial metric value to find the best model (default 0.0)
        eval_iou_thresh (float): an iou threshold to determine correct predict boxes (default 0.5)
        eval_beta (int): a beta value for f_beta score (default 1)
        model_name (str): a part of filename to save (default 'best_model')
        save_best_ckpt (bool): whether to save the best model (default) 
            or its checkpoint (default False)
        checkpoint (dict) (optional): a checkpoint to continue training (default None)
        log_metrics (bool): whether to log metrics into MLflow (default False)
        register_best_log_model (bool): whether to log and register the best model 
            into MLflow (default False)
        reg_model_name: a model registration name (default 'best_model')
        show_random_best_model_prediction (bool): whether to show a random prediction 
            of the best model (default False).

    Returns:
        a dictionary of training and evaluation results.
    """ 
    print("Device: ", device)    
    start_epoch = 0
    best_epoch_score = init_metric_value
    lr_scheduler = None

    model_params = [p for p in model.parameters() if p.requires_grad]
    # Construct an optimizer
    optimizer = getattr(torch.optim, optimizer_name)(model_params, **optim_params)

    if lr_scheduler_name is not None:
        if lr_scheduler_params is None:
            lr_scheduler_params = {}
        # Construct a learning rate scheduler
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler_name)(optimizer, 
                                                                            **lr_scheduler_params)
    
    if checkpoint is not None:
        # Get state parameters from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_epoch_score = checkpoint[metric_to_find_best_model + '_score'] if metric_to_find_best_model else 0.0

    model.to(device)

    for epoch in range(1, epochs+1):
        current_epoch = start_epoch + epoch
        print(f"EPOCH [{current_epoch}/{start_epoch + epochs}]: ")

        # Training step
        print("TRAIN:")
        train_res = train_one_epoch(train_dataloader, model, optimizer, device)
        print("  epoch loss: {0}:\n    {1}".format(train_res['epoch_loss'], 
                                                   train_res['epoch_dict_losses']))

        if lr_scheduler is not None:
            lr_scheduler.step()        
        
        # Evaluation step
        print("EVAL:")
        eval_res = eval_one_epoch(val_dataloader, model, eval_iou_thresh, eval_beta, device)
        print("\n  epoch scores: {}".format(eval_res['epoch_scores'])) 
        
        if metric_to_find_best_model:
            # Save a model with the maximum epoch score
            if best_epoch_score < eval_res['epoch_scores'][metric_to_find_best_model]:
                best_epoch_score = eval_res['epoch_scores'][metric_to_find_best_model]
                ckpt_dict = None
                filename = model_name + f'_best_{metric_to_find_best_model}_{eval_beta}_weights'
                
                if register_best_log_model:
                    try:
                        # Log and register the best model into MLflow
                        mlflow.pytorch.log_model(model, filename, registered_model_name=reg_model_name, 
                                                await_registration_for=40, 
                                                pip_requirements=[f'torch={torch.__version__}', 
                                                                  f'torchvision={torchvision.__version__}'])
                    except NameError: 
                        print("[WARNING]: The Model cannot be registered! -- MLflow module is not imported.")

                if save_best_ckpt:
                    ckpt_dict = {'epoch': current_epoch,            
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 metric_to_find_best_model + '_score': best_epoch_score}
                    filename += '_ckpt'

                save_model_state(model, filename + '.pt', ckpt_dict)
                print("[INFO]: Model is saved. --- The best {} score: {}".format(
                    metric_to_find_best_model, best_epoch_score))

                with torch.no_grad():
                    if show_random_best_model_prediction:
                        sample_imgs, _ = next(iter(val_dataloader))
                        sample_idx = random.randint(0, len(sample_imgs)-1)
                        preds = eval_res['results'][sample_idx]
                        draw_bboxes_on_image(sample_imgs[sample_idx], preds['boxes'], preds['scores'])
                        del sample_imgs
                        del preds
                                    
            if log_metrics: 
                try:   
                    # Log losses and scores into MLflow       
                    mlflow.log_metric('train_epoch_loss', train_res['epoch_loss'], step=current_epoch)
                    mlflow.log_metrics(train_res['epoch_dict_losses'], step=current_epoch)
                    mlflow.log_metrics(eval_res['epoch_scores'], step=current_epoch)
                    print("[INFO]: Metrics are logged.")
                except NameError: 
                    print("[WARNING]: Metrics cannot be logged! -- MLflow module is not imported.")

        # Free up memory
        gc.collect()
        if str(device) == 'cuda':
            torch.cuda.empty_cache()

        print("-" * 60)

    print("DONE!")
    return {'train_res': train_res,
            'eval_res': eval_res}

def main(project_path):
    """Performs fine-tuning of object detection model."""
    project_path = Path(project_path)

    # Get configurations for training and inference
    with open(project_path / CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Set constants
    TRAIN_EVAL_PARAMS = config['model_training_inference_conf']
    DEVICE = get_device(TRAIN_EVAL_PARAMS['device_cuda'])
    BATCH_SIZE = config['image_dataset_conf']['batch_size']
    NUM_CLASSES = config['object_detection_model']['number_classes']

    # Get dataloaders
    train_dl, val_dl, _ = get_train_val_test_dataloaders(BATCH_SIZE, transform_train_img=True)

    # Get a modified model
    model_params = config['object_detection_model']['load_parameters']    
    faster_rcnn_mob_model = faster_rcnn_mob_model_for_n_classes(NUM_CLASSES, **model_params)

    # Load the best parameters for training
    with open(project_path / config['hyperparameter_optimization']['save_best_parameters_path']) as f:
        best_params = yaml.safe_load(f)
    print("[INFO]: The best training parameters are loaded: \n", best_params)

    optimized_train_params = {
        'optimizer_name': best_params['optimizer'],
        'optim_params': best_params[best_params['optimizer']],
        'lr_scheduler_name': best_params['lr_scheduler'],
        'lr_scheduler_params': best_params[best_params['lr_scheduler']]}

    add_train_params = {'epochs': TRAIN_EVAL_PARAMS['epochs'], 
                        'eval_iou_thresh': TRAIN_EVAL_PARAMS['evaluation_iou_threshold'], 
                        'eval_beta': TRAIN_EVAL_PARAMS['evaluation_beta'],
                        'device': DEVICE}
    
    checkpoint = None
    if TRAIN_EVAL_PARAMS['checkpoint']:
        checkpoint_path = project_path / config['object_detection_model']['save_dir'] / TRAIN_EVAL_PARAMS['checkpoint']
        checkpoint = torch.load(checkpoint_path)

    # Train the model (fine-tuning) and log metrics and parameters into MLflow
    mlruns_path = project_path / config['mlops_tracking_conf']['tracking_dir']
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_registry_uri('sqlite:///{}'.format(mlruns_path / 'model_registry.db'))
    ftm_exp = mlflow.get_experiment_by_name('Fine-Tuning_Model')

    if ftm_exp is not None:
        ftm_exp_id = ftm_exp.experiment_id
    else:  
        ftm_exp_id = mlflow.create_experiment('Fine-Tuning_Model')

    with mlflow.start_run(run_name=config['mlops_tracking_conf']['run_train_name'], 
        experiment_id=ftm_exp_id) as mlft_run:

        mlflow.set_tags({'training_process': 'fine_tuning',
                         'model_name': config['object_detection_model']['name'],
                         'tools.training': 'PyTorch'})

        # Run model training cycles
        _ = run_train(train_dl, val_dl, faster_rcnn_mob_model, 
                      save_best_model_path=config['object_detection_model']['save_dir'],
                      metric_to_find_best_model=TRAIN_EVAL_PARAMS['metric_to_find_best'],
                      init_metric_value=0.5, log_metrics=True, save_best_ckpt=True, 
                      model_name=TRAIN_EVAL_PARAMS['model_name'], register_best_log_model=True,
                      reg_model_name=config['object_detection_model']['best_faster_rcnn_mob'],
                      show_random_best_model_prediction=True, checkpoint=checkpoint,
                      **optimized_train_params, **add_train_params)

        # Log the parameters into MLflow
        mlflow.log_params(model_params)
        mlflow.log_params({'seed': SEED,
                           'batch_size': BATCH_SIZE,
                           'num_classes': NUM_CLASSES})
        mlflow.log_params(add_train_params)

        for params in optimized_train_params: 
            if params in ['optim_params', 'lr_scheduler_params']:
                mlflow.log_params(optimized_train_params[params])
            else:
                mlflow.log_param(params, optimized_train_params[params])
        
    print("[INFO]: Parameters are logged.")

if __name__ == '__main__':
    project_path = Path(__file__).parent.parent
    main(project_path)