"""This module evaluates an object detection model on test data."""

import logging
import random
from pathlib import Path

import cv2
import mlflow
import numpy as np
import pandas as pd
import torch

from data.image_dataloader import create_dataloaders
from train.train_inference_fns import eval_one_epoch, predict
from utils import get_config_yml, get_device

logging.basicConfig(level=logging.INFO, filename='app.log',
                    format="[%(levelname)s]: %(message)s")

# Set partial reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def main(project_path, config, show_random_predict=False):
    """Evaluate an object detection model on test data,
    display a random prediction if show_random_predict is True,
    and return a value of a metric used to find the best model.
    """
    img_data_paths = config['image_data_paths']
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
    imgs_path, test_csv_path, bbox_csv_path = [
        project_path / fpath for fpath in [img_data_paths['images'],
                                           img_data_paths['test_csv_file'],
                                           img_data_paths['bboxes_csv_file']]]
    batch_size = config['image_dataset_conf']['batch_size']
    test_dl = create_dataloaders(imgs_path, test_csv_path, bbox_csv_path, batch_size)
    test_res = eval_one_epoch(test_dl, best_faster_rcnn_mob_model,
                              TRAIN_EVAL_PARAMS['evaluation_iou_threshold'],
                              TRAIN_EVAL_PARAMS['evaluation_beta'], DEVICE)
    logging.info(test_res['epoch_scores'])

    if show_random_predict:
        # Save a random test image sample with boxes and scores predictions
        test_imgs_df = pd.read_csv(project_path / config['image_data_paths']['test_csv_file'],
                                   usecols=['Name'])
        test_sample_idx = random.randint(0, test_imgs_df.size - 1)  # nosec
        img_path = project_path / config['image_data_paths']['images']
        test_sample_img = cv2.cvtColor(
            cv2.imread(str(img_path / test_imgs_df.iloc[test_sample_idx].Name)),
            cv2.COLOR_BGR2RGB)
        save_output_path = '/'.join([
            config['model_training_inference_conf']['save_random_best_model_output_dir'],
            f'test_outs/sample_idx_{test_sample_idx}.jpg'])
        _ = predict(test_sample_img, best_faster_rcnn_mob_model, show_scores=True,
                    save_predict_path=project_path / save_output_path)

    return test_res['epoch_scores'][TRAIN_EVAL_PARAMS['metric_to_find_best']]


if __name__ == '__main__':
    project_path = Path.cwd()
    config = get_config_yml()
    mlflow.set_tracking_uri('sqlite:///mlruns/mlruns.db')
    _ = main(project_path, config, show_random_predict=True)
