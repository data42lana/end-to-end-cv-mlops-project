"""This module updates registered model version stages in MLflow."""

import logging
from pathlib import Path

import mlflow

from utils import get_config_yml


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
                m = mlclient.transition_model_version_stage(
                        name=registered_model_name,
                        version=m.version,
                        stage='Production')
        else:
            if m.current_stage == 'Production':
                m = mlclient.transition_model_version_stage(
                        name=registered_model_name,
                        version=m.version,
                        stage='Archived')

    # View updated model version stages
    for m in mlclient.get_latest_versions(registered_model_name):
        logging.info("Updated model version stages: ")
        logging.info(f"{m.name}: version: {m.version}, current stage: {m.current_stage}")

def main(project_path, config):
    """Update version stages for a registered model specified in a configuration file."""
    logging.basicConfig(level=logging.INFO, filename='app.log',
                        format="[%(levelname)s]: %(message)s")
    
    client = mlflow.MlflowClient()
    update_registered_model_version_stages(client, config['object_detection_model']['registered_name'])
    logging.info("Stages are updated.")

if __name__ == '__main__':
    project_path = Path.cwd()
    config = get_config_yml()
    mlflow.set_tracking_uri('sqlite:///mlruns/mlruns.db')
    main(project_path, config)