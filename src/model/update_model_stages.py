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
    prod_run_id = 0
    for m in mlclient.get_latest_versions(registered_model_name):
        logging.info("Updated model version stages: ")
        logging.info(f"{m.name}: version: {m.version}, current stage: {m.current_stage}")

        if m.current_stage == 'Production':
            prod_run_id = m.run_id

    return prod_run_id


def production_model_metric_history_plot(metric_name, mlclient,
                                         registered_model_name, save_path=''):
    """Create metric plots for a production stage models and save them
    if save_path is specified.
    """
    production_model_info = mlclient.get_latest_versions(registered_model_name,
                                                         stages=['Production'])
    prod_metric_plots = []
    for prod_info in production_model_info:
        run_id = prod_info.run_id
        metric_history = mlclient.get_metric_history(run_id, metric_name)

        if metric_name == 'f_beta':
            prod_run_params = mlclient.get_run(run_id).data.params
            metric_name += '_{}'.format(prod_run_params.get('eval_beta', 1))

        metric_step_values = collate_batch([(mh.step, mh.value) for mh in metric_history])
        fig = plt.figure(figsize=(10, 6))
        plt.plot(*metric_step_values, color='blue' if 'loss' in metric_name else 'orange')
        plt.xlabel('epochs')
        plt.ylabel(metric_name)
        plt.title(f"{metric_name.capitalize()} Plot")
        prod_metric_plots.append(fig)

        if save_path:
            save_path = Path(save_path) / 'plots'
            save_path.mkdir(exist_ok=True)
            plt.savefig(save_path / f'{metric_name}.jpg')
            plt.close()
            logging.info("Metric plots of a production stage model are saved.")

    return prod_metric_plots


def main(project_path, config, save_metric_plots=False):
    """Update version stages for a registered model specified
    in a configuration file and create metric plots for production stage model.
    """
    registered_model_name = config['object_detection_model']['registered_name']
    client = mlflow.MlflowClient()
    production_run_id = update_registered_model_version_stages(client, registered_model_name)
    logging.info("Stages are updated.")

    mltracking_conf = config['mlflow_tracking_conf']

    if 'metrics_to_plot' in mltracking_conf:
        metric_plots = []

        for metric in mltracking_conf['metrics_to_plot']:
            metric_plots.append(production_model_metric_history_plot(
                metric, client, registered_model_name,
                save_path=project_path if save_metric_plots else ''))

        return production_run_id, metric_plots
    else:
        return production_run_id


if __name__ == '__main__':
    project_path = Path.cwd()
    config = get_config_yml()
    mlflow.set_tracking_uri('sqlite:///mlruns/mlruns.db')
    _ = main(project_path, config, save_metric_plots=True)
