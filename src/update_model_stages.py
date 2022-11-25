"""This module updates registered model version stages in MLflow."""

from pathlib import Path
import logging

import yaml
import mlflow # Model Registry

CONFIG_PATH = 'configs/config.yaml'

def update_registered_model_version_stages(registered_model_name):
    """Sets a stage to 'Production' for the latest version of model, and 'Archived' 
    if the current stage is 'Production' but the version is not the latest.
    """
    # Get information about a registered model
    client = mlflow.MlflowClient()
    model_registry_info = client.get_latest_versions(registered_model_name)
    model_latest_version = max([m.version for m in model_registry_info])

    # Update model version stages
    for m in model_registry_info:    
        if m.version == model_latest_version:
            if m.current_stage == 'Production':
                continue
            else:
                m = client.transition_model_version_stage(
                        name=registered_model_name,
                        version=m.version,
                        stage='Production')
        else:
            if m.current_stage == 'Production':
                m = client.transition_model_version_stage(
                        name=registered_model_name,
                        version=m.version,
                        stage='Archived')

    # View updated model version stages
    for m in client.get_latest_versions(registered_model_name):
        logging.info("Updated model version stages: ")
        logging.info(f"{m.name}: version: {m.version}, current stage: {m.current_stage}")

def main(project_path):
    """Updates version stages for a registered model specified in a configuration file."""
    project_path = Path(project_path)
    logging.basicConfig(level=logging.INFO)

    with open(project_path / CONFIG_PATH) as f:
        config = yaml.safe_load(f)  
        
    update_registered_model_version_stages(config['object_detection_model']['registered_name'])
    logging.info("Stages are updated.")

if __name__ == '__main__':
    project_path = Path(__file__).parent.parent
    main(project_path)