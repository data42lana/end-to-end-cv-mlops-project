"""This module evaluates and makes prediction of the latest version
of a registered model on test data.
"""

import json
import logging
import random
from pathlib import Path

import mlflow
import numpy as np
import torch

from src.data.image_dataloader import create_dataloaders
from src.train.train_inference_fns import eval_one_epoch, predict
from src.utils import (get_device, get_latest_registared_pytorch_model,
                       get_param_config_yaml, get_random_img_with_info)

logging.basicConfig(level=logging.INFO, filename='app.log',
                    format="[%(levelname)s]: %(message)s")

# Set partial reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def main(project_path, param_config, get_random_prediction=False,
         compare_with_production_model=False):
    """Evaluate the latest version of a registered model on test data, and
    compare with a production model if compare_with_production_model is True,
    and save and return the evaluation result.
    """
    IMG_DATA_PATHS = param_config['image_data_paths']
    TRAIN_EVAL_PARAMS = param_config['model_training_inference_conf']
    DEVICE = get_device(TRAIN_EVAL_PARAMS['device_cuda'])

    # Load the latest version of a model from the MLflow registry
    client = mlflow.MlflowClient()
    reg_model_name = param_config['object_detection_model']['registered_name']
    latest_reg_model = get_latest_registared_pytorch_model(client, reg_model_name)

    # Evaluate the model on test data
    imgs_path, test_csv_path, bbox_csv_path = [
        project_path / fpath for fpath in [IMG_DATA_PATHS['images'],
                                           IMG_DATA_PATHS['test_csv_file'],
                                           IMG_DATA_PATHS['bboxes_csv_file']]]
    batch_size = param_config['image_dataset_conf']['batch_size']
    test_dl = create_dataloaders(imgs_path, test_csv_path, bbox_csv_path, batch_size)
    test_eval_params = {'dataloader': test_dl,
                        'iou_thresh': TRAIN_EVAL_PARAMS['evaluation_iou_threshold'],
                        'beta': TRAIN_EVAL_PARAMS['evaluation_beta'],
                        'device': DEVICE}
    test_eval_res = eval_one_epoch(model=latest_reg_model, **test_eval_params)

    # Get a value of the metric used to find the best model
    test_score_name = TRAIN_EVAL_PARAMS['metric_to_find_best']
    test_score = test_eval_res['epoch_scores'][test_score_name]
    if test_score_name == 'f_beta':
        test_score_name += '_{}'.format(TRAIN_EVAL_PARAMS['evaluation_beta'])
    test_res = {'test_score_value': test_score, 'test_score_name': test_score_name}

    # Compare the model with the latest version of a production model
    if compare_with_production_model:
        prod_reg_model = get_latest_registared_pytorch_model(client, reg_model_name,
                                                             stages=['Production'])
        prod_score = (eval_one_epoch(model=prod_reg_model,
                                     **test_eval_params)['epoch_scores'][test_score_name]
                      if prod_reg_model else 0)
        test_res['best'] = test_score > prod_score

    # Save the test score in a json file
    save_output_path = project_path.joinpath(
        TRAIN_EVAL_PARAMS['save_model_output_dir'], 'test_outs')
    save_output_path.mkdir(exist_ok=True, parents=True)
    with open(save_output_path / 'test_score.json', 'w') as f:
        json.dump(test_res, f)
    logging.info('Test score is saved!')

    if get_random_prediction:
        # Make a random prediction (boxes and scores) on a test image sample and save it
        img_license_pattern = TRAIN_EVAL_PARAMS['license_pattern_to_select_images']
        random_img = get_random_img_with_info(test_csv_path, imgs_path, img_license_pattern,
                                              random_seed=SEED)

        if random_img:
            test_img, test_img_info = random_img
            save_test_predict_path = save_output_path.joinpath(
                'predict-{}'.format(test_img_info['Name']))
            test_pred_res = predict(test_img, latest_reg_model, show_scores=True,
                                    save_predict_path=save_test_predict_path)
            logging.info('Test image with predictions is saved!')
            test_pred_res = {'test_predict_number': test_pred_res[0],
                             'test_predict_img': test_pred_res[1],
                             'test_img_info': test_img_info}
            test_res = {**test_res, **test_pred_res}

    return test_res


if __name__ == '__main__':
    project_path = Path.cwd()
    param_config = get_param_config_yaml(project_path)
    mlflow.set_tracking_uri('sqlite:///mlruns/mlruns.db')
    _ = main(project_path, param_config, get_random_prediction=True,
             compare_with_production_model=True)
