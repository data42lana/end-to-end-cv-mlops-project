"""This module generates a report as a Markdown file containing
a model training result.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.utils import get_param_config_yaml

logging.basicConfig(level=logging.INFO, filename='pipe.log',
                    format="%(asctime)s -- [%(levelname)s]: %(message)s")


def main(project_path, param_config):
    """Generate a report as an .md file containing a test score and
    metric plots and an image with predictions if the score is the best.
    """
    # Set a report title
    report_content = ["# Model Training Result Report\n"]
    data_for_report_dir = param_config['model_training_inference_conf']['save_model_output_dir']

    # Add a test score
    test_output_dir = '/'.join([data_for_report_dir, 'test_outs'])
    test_score_path = project_path / test_output_dir / 'test_score.json'
    with open(test_score_path) as f:
        test_score = json.load(f)
    report_content.append("### test {0} score: **{1}**\n".format(
        test_score['test_score_name'], round(test_score['test_score_value'], 2)))

    # Add model name and version
    model_data = test_score['model_uri'].split('/')
    report_content.append("### model: {0}\n ### version: {1}\n".format(
        model_data[1], model_data[2]))

    test_score_is_best = test_score.get('best', False)

    if test_score_is_best:
        # Add training metric plots
        metric_plots_section_content = ["## Metric History Plot(s):\n"]
        metric_plots_path = '/'.join([data_for_report_dir, 'plots'])
        for plot_path in (project_path / metric_plots_path).iterdir():
            metric_plots_section_content.append(
                "![Metric History Plot](../{0}/{1})".format(metric_plots_path,
                                                            plot_path.parts[-1]))
        report_content += metric_plots_section_content

        # Add an image with test prediction result
        img_pred_section_content = ["\n## Model Test Prediction(s):\n"]
        for fpath in (project_path / test_output_dir).iterdir():
            fname = fpath.parts[-1]

            if fname.startswith('predict-'):
                origin_img_name = fname[8:]
                img_pred_section_content.append(
                    "![Image Test Predict](../{0}/{1})".format(test_output_dir, fname))

                # Create a link to the origin image source
                test_df = pd.read_csv(
                    project_path / param_config['image_data_paths']['test_csv_file'])
                test_img_info = test_df[test_df.Name == origin_img_name].squeeze().to_dict()
                photo_author = test_img_info['Author']
                photo_source = test_img_info['Source']
                photo_source_link = 'https://{0}.com'.format(photo_source.lower())
                photo_number = test_img_info['Name'].split('_', maxsplit=1)[0]
                photo_license = test_img_info['License'].split(')')[0].split('(')[1]
                img_pred_section_content.append(
                    "\nPhoto by {0} on [{1}]({2}). No {3}. License: {4}"
                    "*The photo modified: boxes and scores drawn*.".format(
                        photo_author, photo_source, photo_source_link,
                        photo_number, photo_license))

        report_content += img_pred_section_content

    # Save the report to a .md file
    report_path = project_path / 'reports/model_report.md'
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(''.join(report_content))
    logging.info("Model Training Result Report is saved!")


if __name__ == '__main__':
    project_path = Path.cwd()
    param_config = get_param_config_yaml(project_path)
    main(project_path, param_config)
