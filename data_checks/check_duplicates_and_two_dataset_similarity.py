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

def main(project_path, check_data_type, data_check_dir):
    """Check data for duplicates and similarity between two datasets."""
    project_path = Path(project_path)
    logging.basicConfig(level=logging.INFO)

    # Get image data paths from a configuration file
    config = get_config_yml(project_path)
    img_data_paths = config['image_data_paths']
    
    # Track total check status
    check_passed_conditions = []

    if check_data_type == 'raw':
        # Check data for duplicates
        bbox_df = pd.read_csv(project_path / img_data_paths['bboxes_csv_file'])
        bbox_ds = Dataset(bbox_df, cat_features=['label_name'])
        data_duplicates_check_result = DataDuplicates().run(bbox_ds)
        check_passed_conditions.append(data_duplicates_check_result.passed_conditions())
        # Save check results
        fpath = data_check_dir / f'{check_data_type}_duplicates_check_results.html'
        data_duplicates_check_result.save_as_html(fpath)
        logging.info("Results of checking for duplicates are saved.")

    elif check_data_type == 'prepared':
        # Check similarity between training and test datasets
        train_df, test_df = [
            pd.read_csv(project_path / img_data_paths[csv_file] for csv_file in ['train_csv_file',
                                                                                 'test_csv_file'])]

        tt_cat_features=['Source', 'License']
        train_ds, test_ds = [
            Dataset(df, cat_features=tt_cat_features, index_name='Name') for df in [train_df, test_df]]

        dsc_check = (DatasetsSizeComparison()
                        .add_condition_test_train_size_ratio_greater_than(0.25)
                        .add_condition_train_dataset_greater_or_equal_test())
        ttsm_check = (TrainTestSamplesMix()
                        .add_condition_duplicates_ratio_less_or_equal(0))
        ittl_check = IndexTrainTestLeakage().add_condition_ratio_less_or_equal(0)
        ttfd_info_check = TrainTestFeatureDrift(ignore_columns=['Name', 'Source', 'License'])

        # Train Test Validation
        # Create a custom train test validation suite and run it
        train_test_check_result = check_two_datasets(train_ds, test_ds, 'Train Test Validation Suite', 
                                                     checks=[dsc_check, ttsm_check, 
                                                             ittl_check, ttfd_info_check])
        check_passed_conditions.append(train_test_check_result.passed_conditions())
        # Save check results
        fpath = data_check_dir / f'{check_data_type}_train_test_similarity_check_results.html'
        train_test_check_result.save_as_html(fpath)
        logging.info("Train Test validation results are saved.")

        # Check datasets for Author Group Leakage
        author_train_ds, author_test_ds = [
            Dataset(df['Author'], cat_features=['Author']) for df in (train_df, test_df)]

        # Detect authors in both the training and test data
        author_group_leakage_check_result = check_two_datasets(author_train_ds, author_test_ds, 
                                                               'Train Test Author Group Leakage Suite', 
                                                               checks=[ttsm_check])
        check_passed_conditions.append(author_group_leakage_check_result.passed_conditions())
        # Save check results
        fpath = data_check_dir / f'{check_data_type}_train_test_author_group_leakage_check_results.html'
        author_group_leakage_check_result.save_as_html(fpath)
        logging.info("Author Group Leakage check results are saved.")

    elif check_data_type == 'new':
        # Check similarity between new and old data
        # Check a new image info dataset         
        new_img_data_paths = config['new_image_data_paths']
        info_df, new_info_df = [
            pd.read_csv(project_path / data_path['info_csv_file'] for data_path in [img_data_paths, 
                                                                                    new_img_data_paths])]
        tt_cat_features=['Source', 'License']
        info_ds, new_info_ds = [
            Dataset(df, cat_features=tt_cat_features, index_name='Name') for df in [info_df, new_info_df]]

        ttsm_check = (TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0))
        ittl_check = IndexTrainTestLeakage().add_condition_ratio_less_or_equal(0)
        ttfd_info_check = TrainTestFeatureDrift(ignore_columns=['Name', 'Source', 'License'])

        # Create a custom new info dataset validation suite
        new_info_ds_check_result = check_two_datasets(info_ds, new_info_ds, 
                                                      'New Info Dataset Validation Suite', 
                                                       checks=[ttsm_check, ittl_check, ttfd_info_check])
        check_passed_conditions.append(new_info_ds_check_result.passed_conditions())
        # Save check results
        fpath = data_check_dir / f'{check_data_type}_old_info_check_results.html'
        new_info_ds_check_result.save_as_html(fpath)
        logging.info("New info dataset check results are saved.")

        # Check a new bbox dataset
        bbox_df, new_bbox_df = [
            pd.read_csv(project_path / data_path['bboxes_csv_file'] for data_path in [img_data_paths,
                                                                                      new_img_data_paths])]

        bbox_ds, new_bbox_ds = [
            Dataset(df, cat_features=['label_name']) for df in [bbox_df, new_bbox_df]]

        # Create a custom new bbox dataset validation suite
        new_bbox_ds_check_result = check_two_datasets(
            bbox_ds, new_bbox_ds, 'New Bbox Dataset Validation Suite', 
            checks=[ttsm_check, TrainTestFeatureDrift(ignore_columns=['label_name', 'image_name'])])
        check_passed_conditions.append(new_bbox_ds_check_result.passed_conditions())
        # Save check results
        fpath = data_check_dir / f'{check_data_type}_old_bbox_check_results.html'
        new_bbox_ds_check_result.save_as_html(fpath)
        logging.info("New bbox dataset check results are saved.")

    return bool(sum(check_passed_conditions))

if __name__ == '__main__':
    data_check_dir = Path(__file__).parent
    project_path = data_check_dir.parent
    data_type_parser = argparse.ArgumentParser('Image data csv file check script.', 
                                               parents=[get_data_type_arg_parser()])
    img_data_type = data_type_parser.parse_args()
    check_passed = main(project_path, img_data_type.check_data_type, data_check_dir)

    if not check_passed:
            logging.warning(f"Checking for duplicates and similarity for the {img_data_type} data failed.")