"""This module updates registered model version stages in MLflow and
save metric plots for a production stage model.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow

from utils import collate_batch, get_config_yml

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
    for m in mlclient.get_latest_versions(registered_model_name):
        logging.info("Updated model version stages: ")
        logging.info(f"{m.name}: version: {m.version}, current stage: {m.current_stage}")


def save_production_model_metric_history_plots(metric_names, mlclient,
                                               registered_model_name, save_path):
    """Create and save metric plots for a production stage model."""
    save_path = Path(save_path) / 'plots'
    save_path.mkdir(exist_ok=True)

    production_model_info = mlclient.get_latest_versions(registered_model_name,
                                                         stages=['Production'])

    for prod_info in production_model_info:
        run_id = prod_info.run_id
        for metric in metric_names:
            metric_history = mlclient.get_metric_history(run_id, metric)
            metric_step_values = collate_batch([(mh.step, mh.value) for mh in metric_history])
            plt.figure(figsize=(6, 6))
            plt.plot(*metric_step_values, color='orange')
            plt.xlabel('epochs')
            plt.title(f"{metric.capitalize()} Plot")
            plt.savefig(save_path / f'{metric}.jpg')
            plt.close()


def main(project_path, config):
    """Update version stages for a registered model specified
    in a configuration file and create metric plots for production stage model.
    """
    registered_model_name = config['object_detection_model']['registered_name']
    client = mlflow.MlflowClient()
    update_registered_model_version_stages(client, registered_model_name)
    logging.info("Stages are updated.")

    metrics = config['mlflow_tracking_conf']['metrics_to_plot']
    save_production_model_metric_history_plots(metrics, client,
                                               registered_model_name, project_path)
    logging.info("Metric plots of a production stage model are saved.")


if __name__ == '__main__':
    project_path = Path.cwd()
    config = get_config_yml()
    mlflow.set_tracking_uri('sqlite:///mlruns/mlruns.db')
    main(project_path, config)
