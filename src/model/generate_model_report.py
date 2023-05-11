"""This module generates the report as a Markdown file containing
a model training pipeline result.
"""

import json
import logging
import os
from pathlib import Path

import mlflow
import pandas as pd

from src.utils import (get_current_stage_of_registered_model_version,
                       get_number_of_csv_rows, get_param_config_yaml)

logging.basicConfig(level=logging.INFO, filename='pipe.log',
                    format="%(asctime)s -- [%(levelname)s]: %(message)s")


def fill_in_report_subsection(subsection_title, subsection_content_dict):
    """Add content to a report subsection."""
    subsection = [f"\n{subsection_title}\n"]
    subsection += [f"*{name}:* {val}\n\n" for name, val in subsection_content_dict.items()]
    return subsection


def main(project_path, param_config):
    """Generate the report as a .md file based on the latest training pipeline outputs
    and containing a test score, a random image with predictions,
    and training details with metric plots of model if it is in production.
    """
    # Get the latest training pipeline result from an output dir
    output_dir = param_config['model_training_inference_conf']['save_model_output_dir']
    test_output_dir = '/'.join([output_dir, 'test_outs'])
    test_score_path = project_path / test_output_dir / 'test_score.json'

    with open(test_score_path) as f:
        test_score = json.load(f)

    _, model_name, model_version = test_score['model_uri'].split('/')

    # Get the current stage of the latest model version
    client = mlflow.MlflowClient()
    current_model_stage = get_current_stage_of_registered_model_version(
        client, model_name, int(model_version))

    # Get test data
    test_df = pd.read_csv(
        project_path / param_config['image_data_paths']['test_csv_file'])

    # Set a report title
    report_content = ["# Model Training Pipeline Result Report\n"]

    # Add model information
    model_info = {"Registered Name": f"**{model_name}**",
                  "Version": f"**{model_version}**",
                  "Stage": f"**{current_model_stage}**"}
    report_content += fill_in_report_subsection("## Model Information", model_info)

    # Add model performance on test data
    model_test_perform = {
        test_score['test_score_name'].capitalize(): "**{}**".format(
            round(test_score['test_score_value'], 2)),
        "Test Dataset Size": test_df.shape[0],
        "Example": ""}
    report_content += fill_in_report_subsection("### Performance on Test Data",
                                                model_test_perform)

    # Add a random image with its test prediction result to the "Example" subsection
    img_pred_example_subsection_content = []
    for fpath in (project_path / test_output_dir).iterdir():
        fname = fpath.parts[-1]

        if fname.startswith('predict-'):
            origin_img_name = fname[8:]
            img_pred_example_subsection_content.append(
                "![Image Test Predict](../{0}/{1})".format(test_output_dir, fname))

            # Create a link to the original image source
            test_img_info = test_df[test_df.Name == origin_img_name].squeeze().to_dict()
            photo_author = test_img_info['Author']
            photo_source = test_img_info['Source']
            photo_source_link = 'https://{0}.com'.format(photo_source.lower())
            photo_number = test_img_info['Name'].split('_', maxsplit=1)[0]
            photo_license = test_img_info['License'].split(')')[0].split('(')[1]
            img_pred_example_subsection_content.append(
                "\nPhoto by {0} on [{1}]({2}). No {3}. License: {4} "
                "*The photo modified: boxes and scores drawn*.".format(
                    photo_author, photo_source, photo_source_link,
                    photo_number, photo_license))

    report_content += img_pred_example_subsection_content

    # Expand the report for the model if it is in production
    if current_model_stage == 'Production':
        # Add model training details
        train_size = get_number_of_csv_rows(
            project_path.joinpath(param_config['image_data_paths']['train_csv_file']),
            read_column='Name')
        load_params = param_config['object_detection_model']['load_parameters']
        train_backbone_layers = load_params.get('trainable_backbone_layers', 3)
        max_number_detections = load_params.get('box_detections_per_img', 100)
        model_train_details = {
            "Training Dataset Size": train_size,
            "Number of Trained Backbone Layers": train_backbone_layers,
            "Maximum Number of House Sparrows (that can be detected in one image)": max_number_detections}  # noqa: B950
        report_content += fill_in_report_subsection("## Model Training Details",
                                                    model_train_details)

        # Add training metric history plots
        metric_plots_subsection_content = ["### Metric History Plot(s):\n"]
        metric_plots_path = project_path.joinpath(output_dir, 'plots/metrics')
        for plot_path in metric_plots_path.iterdir():
            metric_plots_subsection_content.append(
                "![Metric History Plot](../{})".format(plot_path.relative_to(project_path)))
        report_content += metric_plots_subsection_content

    # Save the completed report to a .md file
    report_path = project_path / 'reports/model_report.md'
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(''.join(report_content))
    logging.info("Model Training Pipeline Result Report is saved!")


if __name__ == '__main__':
    project_path = Path.cwd()
    param_config = get_param_config_yaml(project_path)
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI',
                            param_config['mlflow_tracking_conf']['mltracking_uri']))
    main(project_path, param_config)
