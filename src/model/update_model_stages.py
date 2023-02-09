"""This module updates registered model version stages in MLflow and
save metric plots for a production stage model.
"""

import argparse
import json
import logging
from pathlib import Path

import mlflow

from utils import get_param_config_yaml, production_model_metric_history_plot

logging.basicConfig(level=logging.INFO, filename='app.log',
                    format="[%(levelname)s]: %(message)s")


def update_registered_model_version_stages(mlclient, registered_model_name):
    """Set a stage to 'Production' for the latest version of a model, and 'Archived'
    if the current stage is 'Production' but the version is not the latest.
    """
    # Get information about a registered model
    model_registry_info = mlclient.get_latest_versions(registered_model_name)
    model_latest_version = max([m.version for m in model_registry_info])

    # Update model version stages
    for m in model_registry_info:
        if m.version == model_latest_version:
            if m.current_stage == 'Production':
                continue
            else:
                m = mlclient.transition_model_version_stage(name=registered_model_name,
                                                            version=m.version,
                                                            stage='Production')
        else:
            if m.current_stage == 'Production':
                m = mlclient.transition_model_version_stage(name=registered_model_name,
                                                            version=m.version,
                                                            stage='Archived')

    # View updated model version stages
    prod_run_id = 0
    for m in mlclient.get_latest_versions(registered_model_name):
        logging.info("Updated model version stages: ")
        logging.info(f"{m.name}: version: {m.version}, current stage: {m.current_stage}")

        if m.current_stage == 'Production':
            prod_run_id = m.run_id

    return prod_run_id


def main(project_path, param_config, save_metric_plots=False):
    """Update version stages for a registered model, return a run id,
    and create and return metric plots for a model with 'Production' stage.
    """
    registered_model_name = param_config['object_detection_model']['registered_name']
    client = mlflow.MlflowClient()
    production_run_id = update_registered_model_version_stages(client, registered_model_name)
    logging.info("Stages are updated.")

    mltraining_conf = param_config['model_training_inference_conf']
    save_path = (project_path.joinpath(mltraining_conf['save_model_output_dir'])
                 if save_metric_plots else None)
    metric_plots = []

    for metric in mltraining_conf['metrics_to_plot']:
        metric_plots.append(production_model_metric_history_plot(metric, client,
                                                                 registered_model_name,
                                                                 save_path=save_path))
    return production_run_id, metric_plots


if __name__ == '__main__':
    project_path = Path.cwd()
    param_config = get_param_config_yaml(project_path)
    mlflow.set_tracking_uri('sqlite:///mlruns/mlruns.db')

    run_parser = argparse.ArgumentParser(
        description='Specify a condition to run this module.',
        add_help=False)
    run_parser.add_argument(
        '--only_if_test_score_is_best', type=bool,
        default=False, help='whether to run this module only if a test score is the best')

    if run_parser().parse_args().only_if_test_score_is_best:
        test_score_path = project_path.joinpath(
            param_config['model_training_inference_conf']['save_model_output_dir'],
            '/test_outs/test_score.json')
        with open(test_score_path) as f:
            test_score_is_best = json.load(f)['best']

        if test_score_is_best:
            main(project_path, param_config, save_metric_plots=True)
        else:
            logging.info("Stages update did not run: a test score is not the best!")

    else:
        main(project_path, param_config)
