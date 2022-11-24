"""This module evaluates an object detection model on test data."""
    
import random
from pathlib import Path
import logging

import yaml
import numpy as np
import pandas as pd
import cv2
import mlflow # Model Registry
import torch # PyTorch

from train_inference_fns import eval_one_epoch, predict
from image_dataloader import get_train_val_test_dataloaders
from utils import get_device

CONFIG_PATH = 'configs/config.yaml'

# Set partial reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def main(project_path, show_random_predict=False):
    """Evaluates an object detection model on test data 
    and displays a random prediction if show_random_predict is True.
    """
    project_path = Path(project_path)    
    logging.basicConfig(level=logging.INFO)
    
    # Get configurations
    with open(project_path / CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    TRAIN_EVAL_PARAMS = config['model_training_inference_conf']
    DEVICE = get_device(TRAIN_EVAL_PARAMS['device_cuda'])

    # Load the best model from the MLflow registry
    client = mlflow.MlflowClient()
    reg_model_name = config['object_detection_model']['registered_name']
    model_registry_info = client.get_latest_versions(reg_model_name)
    model_latest_version = max([m.version for m in model_registry_info])
    model_uri = 'models:/{}/{}'.format(reg_model_name, model_latest_version)
    best_faster_rcnn_mob_model = mlflow.pytorch.load_model(model_uri)

    # Evaluate the best model on test data
    test_dl = get_train_val_test_dataloaders(config['image_dataset_conf']['batch_size'])[2]
    test_res = eval_one_epoch(test_dl, best_faster_rcnn_mob_model, 
                              TRAIN_EVAL_PARAMS['evaluation_iou_threshold'], 
                              TRAIN_EVAL_PARAMS['evaluation_beta'], DEVICE)
    logging.info(test_res['epoch_scores'])

    if show_random_predict:
        # Display a random test image sample with predict boxes and scores
        test_imgs_df = pd.read_csv(project_path / config['image_data_paths']['test_csv_file'], 
                                   usecols=['Name'])
        test_sample_idx = random.randint(0, test_imgs_df.size-1)
        img_path = project_path / config['image_data_paths']['images']
        test_sample_img = cv2.cvtColor(cv2.imread(str(img_path / test_imgs_df.iloc[test_sample_idx].Name)), 
                                       cv2.COLOR_BGR2RGB)
        _ = predict(test_sample_img, best_faster_rcnn_mob_model, show_scores=True)

if __name__ == '__main__':
    project_path = Path(__file__).parent.parent
    main(project_path, show_random_predict=True)