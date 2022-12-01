""""This module checks data for duplicates and similarity between two datasets."""

import argparse
from pathlib import Path
import logging

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular import Suite
from deepchecks.tabular.checks import (DataDuplicates, DatasetsSizeComparison, TrainTestSamplesMix, 
                                       IndexTrainTestLeakage, TrainTestFeatureDrift)

from utils import get_data_type_arg_parser, get_config_yml

def check_two_datasets(ds1, ds2, suite_name, checks):
    """Create a custom validation suite and return its check result.
    
    Parameters:
    ds1 -- a Dataset object
    ds2 -- a Dataset object
    checks -- a list containing check objects for creating a custom suite
    suite_name -- a name for the created custom suite.
    """
    custom_suite = Suite(suite_name, *checks)
    custom_suite_result = custom_suite.run(ds1, ds2)
    return custom_suite_result

def check_bbox_csv_file_for_duplicates(bbox_file_path):
    """Check a bounding boxes CSV file for duplicates."""
    bbox_df = pd.read_csv(bbox_file_path)
    bbox_ds = Dataset(bbox_df, cat_features=['label_name'])
    check_result = DataDuplicates().run(bbox_ds)
    return check_result

def check_two_datasets_similarity(file_path_1, file_path_2, check_type='train-test', 
                                  check_new_old_bbox_files=False):
    """Check similarity of training and test if check_type='train-test' 
    or new and old if check_type='new-old' datasets (CSV files).
    """
    tt_cat_features=['Source', 'License']
    df1, df2 = [pd.read_csv(csv_file) for csv_file in [file_path_1, file_path_2]]
    ds1, ds2 = [
        Dataset(df, cat_features=tt_cat_features, index_name='Name') for df in [df1, df2]]

    ttsm_check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0)
    ittl_check = IndexTrainTestLeakage().add_condition_ratio_less_or_equal(0)
    ttfd_info_check = TrainTestFeatureDrift(ignore_columns=['Name', 'Source', 'License'])
    check_suite = [ttsm_check, ittl_check, ttfd_info_check]

    if check_type == 'train-test':
        dsc_check = (DatasetsSizeComparison()
                        .add_condition_test_train_size_ratio_greater_than(0.25)
                        .add_condition_train_dataset_greater_or_equal_test())
        check_suite_name = 'Train Test Validation Suite'
        check_suite = [dsc_check] + check_suite
    elif check_type == 'new-old':

        if check_new_old_bbox_files:
            ds1, ds2 = [Dataset(df, cat_features=['label_name']) for df in [df1, df2]]
            check_suite_name = 'New Bbox Dataset Validation Suite'
            check_suite = [ttsm_check, TrainTestFeatureDrift(ignore_columns=['label_name', 'image_name'])]
        else:
            check_suite_name = 'New Info Dataset Validation Suite'

    else:
        raise ValueError("check_type must be equal 'train-test' or 'new-old'!")
    
    check_result = check_two_datasets(ds1, ds2, check_suite_name, checks=check_suite)

    return check_result

def check_train_test_author_group_leakage(train_file_path, test_file_path):
    """Check training and test datasets (CSV files) for author group leakage."""
    train_df, test_df = [pd.read_csv(csv_file) for csv_file in [train_file_path, test_file_path]]
    author_train_ds, author_test_ds = [
        Dataset(df['Author'], cat_features=['Author']) for df in (train_df, test_df)]

    ttsm_check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0)

    check_result = check_two_datasets(author_train_ds, author_test_ds, 
                                      'Train Test Author Group Leakage Suite', 
                                      checks=[ttsm_check])
    return check_result

def main(project_path, check_data_type, data_check_dir):
    """Check data for duplicates and similarity between two datasets."""
    project_path = Path(project_path)
    logging.basicConfig(level=logging.INFO, filename='logs/similarity_checks_log.txt',
                        format="[%(levelname)s]: %(message)s")

    # Get image data paths from a configuration file
    config = get_config_yml(project_path)
    img_data_paths = config['image_data_paths']
    
    # Track total check status
    check_passed_conditions = []
    check_results = []

    fnames = []
    log_msgs = []

    if check_data_type == 'raw':
        # Data Duplicates Check
        data_duplicates_check_result = check_bbox_csv_file_for_duplicates(
            project_path / img_data_paths['bboxes_csv_file'])            
        check_results.append(data_duplicates_check_result)
        check_passed_conditions.append(data_duplicates_check_result.passed_conditions())
        fnames.append(f'{check_data_type}_duplicates')
        log_msgs.append("Results of checking for duplicates are saved.")

    elif check_data_type == 'prepared':
        train_path, test_path = [project_path / img_data_paths[csv_file] for csv_file in ['train_csv_file',
                                                                                          'test_csv_file']]

        # Train Test Validation
        train_test_check_result = check_two_datasets_similarity(train_path, test_path)
        check_results.append(train_test_check_result)
        check_passed_conditions.append(train_test_check_result.passed_conditions())
        fnames.append(f'{check_data_type}_train_test_similarity')
        log_msgs.append("Train Test validation results are saved.")

        # Train Test Author Group Leakage
        author_group_leakage_check_result = check_train_test_author_group_leakage(train_path, test_path)
        check_results.append(author_group_leakage_check_result)
        check_passed_conditions.append(author_group_leakage_check_result.passed_conditions())
        fnames.append(f'{check_data_type}_train_test_author_leakage')
        log_msgs.append("Author Group Leakage check results are saved.")

    elif check_data_type == 'new':
        new_img_data_paths = config['new_image_data_paths']

        # New Info Dataset Check
        info_path, new_info_path = [
            project_path / data_path['info_csv_file'] for data_path in [img_data_paths, 
                                                                        new_img_data_paths]]
        new_info_ds_check_result = check_two_datasets_similarity(info_path, new_info_path, 
                                                                 check_type='new-old')
        check_results.append(new_info_ds_check_result)
        check_passed_conditions.append(new_info_ds_check_result.passed_conditions())
        fnames.append(f'{check_data_type}_old_info')
        log_msgs.append("New info dataset check results are saved.")

        # New Bbox Dataset Check
        bbox_path, new_bbox_path = [
            project_path / data_path['bboxes_csv_file'] for data_path in [img_data_paths, 
                                                                          new_img_data_paths]]
        new_bbox_ds_check_result = check_two_datasets_similarity(bbox_path, new_bbox_path, 
                                                                 check_type='new-old', 
                                                                 check_new_old_bbox_files=True)
        check_results.append(new_bbox_ds_check_result)
        check_passed_conditions.append(new_bbox_ds_check_result.passed_conditions())
        fnames.append(f'{check_data_type}_old_bbox')
        log_msgs.append("New bbox dataset check results are saved.")
    else:
        raise ValueError("check_data_type must be one of 'raw', 'prepared', or 'new'!")
    
    # Save check results
    for check_result, fname, log_msg in zip(check_results, fnames, log_msgs):
        fname += '_check_results.html'
        check_result.save_as_html(data_check_dir / fname)
        logging.info(log_msg)

    return bool(sum(check_passed_conditions))

if __name__ == '__main__':
    data_check_dir = Path(__file__).parent
    project_path = data_check_dir.parent
    data_type_parser = argparse.ArgumentParser('Image data CSV file check script.', 
                                               parents=[get_data_type_arg_parser()])
    img_data_type = data_type_parser.parse_args()

    if img_data_type in ['raw', 'prepared', 'new']:
        check_passed = main(project_path, img_data_type.check_data_type, data_check_dir)

        if not check_passed:
            logging.warning(f"Checking for duplicates or similarity of \
                the {img_data_type} CSV files failed.")

    else:
        logging.warning(f"{img_data_type} data cannot be checked. Choose 'raw', 'prepared', or 'new'.")