"""This module checks the performance of a deployed ML model
and generates a report to debug the model quality decay.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.options import ColorOptions
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (TestColumnDrift, TestColumnValueMean,
                             TestColumnValueMedian, TestShareOfOutRangeValues)

from src.utils import get_number_of_csv_rows, get_param_config_yaml

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, filename='app.log',
                    format="%(asctime)s -- [%(levelname)s]: %(message)s")


def get_selected_monitoring_data(monitoring_data_path, model_name_and_version_parameters,
                                 usecols=None, skiprows=None):
    """Return a pd.Dataframe containing monitoring data for the specified model only.

    Parameters
    ----------
    monitoring_data_path: Path
        A path to data of deployed models for monitoring their performance.
    model_name_and_version_parameters: dict
        Columns containing model name and version and their corresponding value
        to select monitoring data.
    usecols: list, optional
        Columns for the pd.read_csv function parameter of the same name (default None).
    skiprows: list, optional
        Rows for the pd.read_csv function parameter of the same name (default None).
    """
    monitoring_data_df = pd.read_csv(
        monitoring_data_path, usecols=usecols, skiprows=skiprows)
    model_data_sel_conds = [monitoring_data_df[col] == model_name_and_version_parameters[col]
                            for col in model_name_and_version_parameters.keys()]
    return monitoring_data_df.loc[np.logical_and(*model_data_sel_conds)]


def detect_deployed_model_decay_by_scores(reference_monitoring_score_df,
                                          current_monitoring_score_df,
                                          score_column, color_scheme=None,
                                          save_check_results_path=None):
    """Check by a score column if a deployed ML model is decaying and save the results
    if save_check_results_path is specified.
    """
    suite = TestSuite(tests=[TestShareOfOutRangeValues(column_name=score_column, gte=0, lte=1),
                             TestColumnDrift(column_name=score_column),
                             TestColumnValueMean(column_name=score_column),
                             TestColumnValueMedian(column_name=score_column)],
                      options=[color_scheme] if color_scheme is not None else color_scheme)
    suite.run(reference_data=reference_monitoring_score_df,
              current_data=current_monitoring_score_df)
    model_decay_detected = not suite.as_dict()['summary']['all_passed']

    if save_check_results_path:
        save_check_results_path.parent.mkdir(exist_ok=True, parents=True)
        suite.save_html(str(save_check_results_path))

    return model_decay_detected


def generate_deployed_model_report(reference_monitoring_data_df,
                                   current_monitoring_data_df,
                                   column_mapping=None, color_scheme=None,
                                   save_report_path=None):
    """Generate a report containing performance data of a deployed model
    and save it if save_report_path is specified.
    """
    report = Report(metrics=[DataQualityPreset(),
                             DataDriftPreset()],
                    options=[color_scheme] if color_scheme is not None else color_scheme)
    report.run(reference_data=reference_monitoring_data_df,
               current_data=current_monitoring_data_df,
               column_mapping=column_mapping)

    if save_report_path:
        save_report_path.parent.mkdir(exist_ok=True, parents=True)
        report.save_html(str(save_report_path))

    return report.as_dict()


def main(project_path, param_config):
    """Check the performance of a deployed ML model and generates a report
    if the decay of the model is detected.
    """
    MONITORING_CONFIG = param_config['deployed_model_monitoring']
    monitoring_data_path = (
        project_path / MONITORING_CONFIG['save_monitoring_data_path'])

    # Columns for checking and reporting
    MODEL_COLUMNS = ['reg_model_name', 'reg_model_version']

    # Columns for checking
    SCORE_COLUMN = 'bbox_score'
    LABEL_COLUMN = 'labels'

    # Set color scheme options
    color_scheme = ColorOptions()
    color_scheme.primary_color = '#d95a00'  # and color_scheme.current_data_color

    # Rows for checking and reporting
    total_nrows = get_number_of_csv_rows(monitoring_data_path, read_column=LABEL_COLUMN)
    check_nrows = MONITORING_CONFIG['max_total_number_of_records_to_load']
    skip_rows = (lambda x: x in range(0, total_nrows - check_nrows)
                 if total_nrows > check_nrows else None)

    # Get current deployed model parameters
    current_deployed_model_params = get_param_config_yaml(
        project_path, MONITORING_CONFIG['save_deployed_model_params_path'])
    current_model_selection_params = {
        'reg_model_name': current_deployed_model_params['registered_model_name'],
        'reg_model_version': current_deployed_model_params['registered_model_version']}

    # Get box scores only for the model that is currently deployed
    current_deployed_model_score_df = get_selected_monitoring_data(
        monitoring_data_path, current_model_selection_params,
        usecols=[LABEL_COLUMN, SCORE_COLUMN] + MODEL_COLUMNS,
        skiprows=skip_rows)

    # Check if there are records for the model
    if current_deployed_model_score_df.shape[0] == 0:
        raise ValueError(f"Monitoring data is empty: {current_model_selection_params}")

    # Check if the model works correctly
    elif np.any(current_deployed_model_score_df[LABEL_COLUMN] != 1):
        raise ValueError("Labels can only have the value 1!")

    # Check if the model is decaying
    reference_score_df, current_score_df = np.array_split(
        current_deployed_model_score_df, 2, axis=0)
    check_result_path = project_path.joinpath(
        MONITORING_CONFIG['save_monitoring_check_results_dir'],
        'deployed_model_check_results/model_decay_check_results.html')
    deployed_model_decay_detected = detect_deployed_model_decay_by_scores(
        reference_score_df, current_score_df, SCORE_COLUMN, color_scheme, check_result_path)
    logging.info("Model Decay Check results are saved.")

    if deployed_model_decay_detected:
        # Columns for reporting
        BBOX_SIZE_COLUMNS = ['bbox_width', 'bbox_height']
        IMAGE_SIZE_COLUMNS = ['image_width', 'image_height']

        # Get data only for the model that is currently deployed
        current_deployed_model_data_df = get_selected_monitoring_data(
            monitoring_data_path, current_model_selection_params,
            skiprows=skip_rows)

        # Get reference data to reporting
        img_data_paths = param_config['image_data_paths']
        train_img_names = (pd.read_csv(img_data_paths['train_csv_file'], usecols=['Name'])
                             .squeeze().to_list())
        img_bbox_df = pd.read_csv(img_data_paths['bboxes_csv_file'])
        reference_data_df = img_bbox_df.loc[img_bbox_df.image_name.isin(train_img_names),
                                            BBOX_SIZE_COLUMNS + IMAGE_SIZE_COLUMNS]

        # Get current data to reporting
        _, current_data_df = np.array_split(current_deployed_model_data_df, 2, axis=0)
        current_data_df['bbox_width'] = current_data_df['bbox_x1'] + current_data_df['bbox_x2']
        current_data_df['bbox_height'] = current_data_df['bbox_y1'] + current_data_df['bbox_y2']

        # Set column mapping options
        column_mapping = ColumnMapping()
        column_mapping.numerical_features = BBOX_SIZE_COLUMNS + IMAGE_SIZE_COLUMNS

        # Generate a report to debug the model quality decay
        save_report_path = 'reports/deployed_model_performance_report.html'
        _ = generate_deployed_model_report(
            reference_data_df, current_data_df[BBOX_SIZE_COLUMNS + IMAGE_SIZE_COLUMNS],
            column_mapping, color_scheme, save_report_path=project_path / save_report_path)
        logging.info("Deployed Model Performance Report is saved.")


if __name__ == '__main__':
    project_path = Path.cwd()
    param_config = get_param_config_yaml(project_path)
    main(project_path, param_config)
