"""This module checks data for duplicates and similarity between two datasets."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (DataDuplicates, DatasetsSizeComparison,
                                       TrainTestFeatureDrift, TrainTestSamplesMix)

from data_checks.dch_utils import get_data_path_config_yaml, get_data_type_arg_parser


def check_two_datasets(ds1, ds2, suite_name, checks):
    """Create a custom validation suite and return its check result.

    Parameters
    ----------
    ds1: Dataset
        A Dataset object.
    ds2: Dataset
        A Dataset object.
    checks: list
        A list containing check objects for creating a custom suite.
    suite_name: str
        A name for the created custom suite.
    """
    custom_suite = Suite(suite_name, *checks)
    custom_suite_result = custom_suite.run(ds1, ds2)
    return custom_suite_result


def check_bbox_data_for_duplicates(bbox_df):
    """Check a bounding box pd.DataFrame for duplicates."""
    bbox_ds = Dataset(bbox_df, cat_features=['label_name'])
    dd_check = DataDuplicates().add_condition_ratio_less_or_equal(0)
    check_result = dd_check.run(bbox_ds)
    return check_result


def check_two_datasets_similarity(df1, df2, check_type='train-test',
                                  check_bbox_data=False):
    """Check similarity of training and test datasets (pd.DataFrames)
    if check_type='train-test' or new and old ones if check_type='new-old'.
    """
    if check_bbox_data:
        index_name = None
        cat_features = ['label_name']
        ignore_columns = ['label_name', 'image_name']
        check_suite_name = 'Bbox Dataset Validation Suite'
    else:
        index_name = 'Name'
        cat_features = ['Source', 'License']
        ignore_columns = ['Name', 'Source', 'License']
        check_suite_name = 'Info Dataset Validation Suite'

    ds1, ds2 = [Dataset(df, cat_features=cat_features, index_name=index_name)
                for df in [df1, df2]]
    ttsm_check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0)
    ttfd_check = TrainTestFeatureDrift(ignore_columns=ignore_columns)
    check_suite = [ttsm_check, ttfd_check]

    if check_type == 'train-test':
        dsc_check = (DatasetsSizeComparison()
                        .add_condition_test_train_size_ratio_greater_than(0.25)  # noqa: E127
                        .add_condition_train_dataset_greater_or_equal_test())  # noqa: E127
        check_suite_name = 'Train Test Validation Suite'
        check_suite = [dsc_check] + check_suite
    elif check_type == 'new-old':
        check_suite_name = 'New ' + check_suite_name
    else:
        raise ValueError("check_type must be equal 'train-test' or 'new-old'!")

    check_suite_result = check_two_datasets(ds1, ds2, check_suite_name, checks=check_suite)

    return check_suite_result


def check_train_test_author_group_leakage(train_df, test_df):
    """Check training and test pd.DataFrames for author group leakage."""
    author_train_ds, author_test_ds = [
        Dataset(df['Author'], cat_features=['Author']) for df in (train_df, test_df)]

    ttsm_check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0)

    check_suite_result = check_two_datasets(author_train_ds, author_test_ds,
                                            'Train Test Author Group Leakage Suite',
                                            checks=[ttsm_check])
    return check_suite_result


def main(project_path, data_path_config, check_data_type, data_check_dir):
    """Check data for duplicates and similarity between two datasets."""
    logging.basicConfig(level=logging.INFO, filename='pipe.log',
                        format="%(asctime)s -- [%(levelname)s]: %(message)s")

    # Get image data paths from configurations
    img_data_paths = data_path_config['image_data_paths']

    # Track total check status
    checks_passed = []
    check_results = []

    fnames = []
    log_msgs = []

    if check_data_type == 'raw':
        # Bbox Duplicates Check
        bbox_df = pd.read_csv(project_path / img_data_paths['bboxes_csv_file'])
        data_duplicates_check_result = check_bbox_data_for_duplicates(bbox_df)
        check_results.append(data_duplicates_check_result)
        checks_passed.append(data_duplicates_check_result.passed_conditions())
        fnames.append(f'{check_data_type}_bbox_duplicates')
        log_msgs.append("Results of checking for bbox duplicates are saved.")

    elif check_data_type == 'prepared':
        train_path, test_path = [
            project_path / img_data_paths[csv_file] for csv_file in ['train_csv_file',
                                                                     'test_csv_file']]
        train_df, test_df = [pd.read_csv(csv_file) for csv_file in [train_path, test_path]]

        # Train Test Validation
        train_test_check_result = check_two_datasets_similarity(train_df, test_df,
                                                                check_type='train-test')
        check_results.append(train_test_check_result)
        checks_passed.append(train_test_check_result.passed(fail_if_check_not_run=True))
        fnames.append(f'{check_data_type}_train_test_similarity')
        log_msgs.append("Train Test validation results are saved.")

        # Train Test Author Group Leakage
        author_group_leakage_check_result = check_train_test_author_group_leakage(
            train_df, test_df)
        check_results.append(author_group_leakage_check_result)
        checks_passed.append(author_group_leakage_check_result.passed(fail_if_check_not_run=True))
        fnames.append(f'{check_data_type}_train_test_author_leakage')
        log_msgs.append("Author Group Leakage check results are saved.")

    elif check_data_type == 'new':
        new_img_data_paths = data_path_config['new_image_data_paths']

        # New Info Dataset Check
        info_path, new_info_path = [
            project_path / data_path['info_csv_file'] for data_path in [img_data_paths,
                                                                        new_img_data_paths]]
        info_df, new_info_df = [
            pd.read_csv(csv_file) for csv_file in [info_path, new_info_path]]
        new_info_ds_check_result = check_two_datasets_similarity(info_df, new_info_df,
                                                                 check_type='new-old')
        check_results.append(new_info_ds_check_result)
        checks_passed.append(new_info_ds_check_result.passed(fail_if_check_not_run=True))
        fnames.append(f'{check_data_type}_old_info')
        log_msgs.append("New info similarity check results are saved.")

        # New Bbox Dataset Check
        bbox_path, new_bbox_path = [
            project_path / data_path['bboxes_csv_file'] for data_path in [img_data_paths,
                                                                          new_img_data_paths]]
        bbox_df, new_bbox_df = [
            pd.read_csv(csv_file) for csv_file in [bbox_path, new_bbox_path]]
        new_bbox_ds_check_result = check_two_datasets_similarity(bbox_df, new_bbox_df,
                                                                 check_type='new-old',
                                                                 check_bbox_data=True)
        check_results.append(new_bbox_ds_check_result)
        checks_passed.append(new_bbox_ds_check_result.passed(fail_if_check_not_run=True))
        fnames.append(f'{check_data_type}_old_bbox')
        log_msgs.append("New bbox similarity check results are saved.")
    else:
        raise ValueError("check_data_type must be one of 'raw', 'prepared', or 'new'!")

    # Save check results
    for check_result, fname, log_msg in zip(check_results, fnames, log_msgs):
        fname += '_check_results.html'
        fpath = data_check_dir / 'data_check_results' / fname
        if fpath.exists():
            fpath.unlink(missing_ok=True)
        else:
            fpath.parent.mkdir(parents=True, exist_ok=True)
        check_result.save_as_html(str(fpath))
        logging.info(log_msg)

    return np.all(checks_passed)


if __name__ == '__main__':
    project_path = Path.cwd()
    data_path_config = get_data_path_config_yaml(project_path)
    data_check_dir = Path(__file__).parent
    img_data_type = get_data_type_arg_parser().parse_args().check_data_type

    if img_data_type in ['raw', 'prepared', 'new']:
        check_passed = main(project_path, data_path_config, img_data_type, data_check_dir)

        if not check_passed:
            raise ValueError(
                f"Checking for duplicates/similarity of the {img_data_type} CSV files failed.")  # noqa: B950

    else:
        raise ValueError(
            f"{img_data_type} data cannot be checked: choose 'raw', 'prepared', or 'new'.")
