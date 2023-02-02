"""This module evaluates and makes prediction of the latest version
of a registered model on test data.
"""

import logging
import random
from pathlib import Path

import mlflow
import numpy as np
import torch

from data.image_dataloader import create_dataloaders
from train.train_inference_fns import eval_one_epoch, predict
from utils import (get_config_yml, get_device, get_latest_registared_pytorch_model,
                   get_random_img_with_info)

logging.basicConfig(level=logging.INFO, filename='app.log',
                    format="[%(levelname)s]: %(message)s")

# Set partial reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def main(project_path, config, get_random_prediction=False):
    """Evaluate the latest version of a registered model on test data,
    and make and save a prediction on a randomly selected test image
    if get_random_predict is True.
    """
    IMG_DATA_PATHS = config['image_data_paths']
    TRAIN_EVAL_PARAMS = config['model_training_inference_conf']
    DEVICE = get_device(TRAIN_EVAL_PARAMS['device_cuda'])

    # Load the latest version of a model from the MLflow registry
    client = mlflow.MlflowClient()
    reg_model_name = config['object_detection_model']['registered_name']
    latest_faster_rcnn_mob_model = get_latest_registared_pytorch_model(client, reg_model_name)

    # Evaluate the model on test data
    imgs_path, test_csv_path, bbox_csv_path = [
        project_path / fpath for fpath in [IMG_DATA_PATHS['images'],
                                           IMG_DATA_PATHS['test_csv_file'],
                                           IMG_DATA_PATHS['bboxes_csv_file']]]
    batch_size = config['image_dataset_conf']['batch_size']
    test_dl = create_dataloaders(imgs_path, test_csv_path, bbox_csv_path, batch_size)
    test_eval_res = eval_one_epoch(test_dl, latest_faster_rcnn_mob_model,
                                   TRAIN_EVAL_PARAMS['evaluation_iou_threshold'],
                                   TRAIN_EVAL_PARAMS['evaluation_beta'], DEVICE)
    test_score = test_eval_res['epoch_scores'][TRAIN_EVAL_PARAMS['metric_to_find_best']]
    test_res = {'test_score': test_score}
    logging.info(test_eval_res['epoch_scores'])

    if get_random_prediction:
        # Make a random prediction (boxes and scores) on a test image sample and save it
        img_license_pattern = TRAIN_EVAL_PARAMS['license_pattern_to_select_images']
        random_img = get_random_img_with_info(test_csv_path, imgs_path, img_license_pattern,
                                              random_seed=SEED)

        if random_img:
            test_img, test_img_info = random_img
            save_output_path = '/'.join([
                TRAIN_EVAL_PARAMS['save_random_best_model_output_dir'],
                'test_outs/sample_img_{}'.format(test_img_info['Name'])])
            test_pred_res = predict(test_img, latest_faster_rcnn_mob_model, show_scores=True,
                                    save_predict_path=project_path / save_output_path)
            test_pred_res = {'test_predict_number': test_pred_res[0],
                             'test_predict_img': test_pred_res[1],
                             'test_img_info': test_img_info}
            test_res = {**test_res, **test_pred_res}

    return test_res


if __name__ == '__main__':
    project_path = Path.cwd()
    config = get_config_yml()
    mlflow.set_tracking_uri('sqlite:///mlruns/mlruns.db')
    _ = main(project_path, config, get_random_prediction=True)
