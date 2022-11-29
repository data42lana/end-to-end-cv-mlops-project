import os # !!!!! 
from pathlib import Path

import numpy as np
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular import Suite
from deepchecks.tabular.checks import (DataDuplicates, DatasetsSizeComparison, TrainTestSamplesMix, 
                                       IndexTrainTestLeakage, TrainTestFeatureDrift)

DATA_PATH = '../data/'
RAW_DATA_PATH, PREPARED_DATA_PATH, NEW_DATA_PATH = [os.path.join(DATA_PATH, fdir) for fdir in ['raw', 'prepared', 'new']]

# ## 2. Single Dataset Checks 

bbox_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'bboxes/bounding_boxes.csv'))

bbox_ds = Dataset(bbox_df, cat_features=['label_name'])

# Run the check and show result
data_duplicates_check_result = DataDuplicates().run(bbox_ds)
data_duplicates_check_result.value

# ## 3. Train Test Dataset Checks

def check_two_datasets(ds1, ds2, suite_name, checks):
    """Create a custom validation suite and displays check result.
    
    Parameters:
    ds1 -- a Dataset object
    ds2 -- a Dataset object
    checks -- a list containing check objects for creating a custom suite
    suite_name -- a name for the created custom suite.
    """
    custom_suite = Suite(suite_name, *checks)
    print(custom_suite)
    custom_suite_result = custom_suite.run(ds1, ds2)
    custom_suite_result.show_in_iframe()    

train_df = pd.read_csv(os.path.join(PREPARED_DATA_PATH, 'train.csv'))
test_df = pd.read_csv(os.path.join(PREPARED_DATA_PATH, 'test.csv'))

tt_cat_features=['Source', 'License']

train_ds = Dataset(train_df, cat_features=tt_cat_features, index_name='Name')
test_ds = Dataset(test_df, cat_features=tt_cat_features, index_name='Name')

dsc_check = (DatasetsSizeComparison()
                 .add_condition_test_train_size_ratio_greater_than(0.25)
                 .add_condition_train_dataset_greater_or_equal_test())
ttsm_check = (TrainTestSamplesMix()
                  .add_condition_duplicates_ratio_less_or_equal(0))
ittl_check = IndexTrainTestLeakage().add_condition_ratio_less_or_equal(0)
ttfd_info_check = TrainTestFeatureDrift(ignore_columns=['Name', 'Source', 'License'])

# ### 3.1. Train Test Validation

# Create a custom train test validation suite and display result
check_two_datasets(train_ds, test_ds, 'Train Test Validation Suite', 
                   checks=[dsc_check, ttsm_check, ittl_check, ttfd_info_check])

# ### 3.2. Author Group Leakage

author_train_ds, author_test_ds = [Dataset(df['Author'], cat_features=['Author']) for df in (train_df, test_df)]

# Detect authors in both the training and test data and display result
check_two_datasets(author_train_ds, author_test_ds, 
                   'Train Test Author Group Leakage Suite', 
                   checks=[ttsm_check])

# ## 4. New Dataset Checks 

# ### 4.1. New Info Dataset Check

info_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'image_info.csv'))
new_info_df = pd.read_csv(os.path.join(NEW_DATA_PATH, 'new_image_info.csv'))

info_ds = Dataset(info_df, cat_features=tt_cat_features, index_name='Name')
new_info_ds = Dataset(new_info_df, cat_features=tt_cat_features, index_name='Name')

# Create a custom new info dataset validation suite and display result
check_two_datasets(info_ds, new_info_ds, 'New Info Dataset Validation Suite', 
                   checks=[ttsm_check, ittl_check, ttfd_info_check])

# ### 4.2. New Bbox Dataset Check

new_bbox_df = pd.read_csv(os.path.join(NEW_DATA_PATH, 'bboxes/new_bounding_boxes.csv'))

new_bbox_ds = Dataset(new_bbox_df, cat_features=['label_name'])

# Create a custom new bbox dataset validation suite and display result
check_two_datasets(bbox_ds, new_bbox_ds, 'New Bbox Dataset Validation Suite', 
                   checks=[ttsm_check, TrainTestFeatureDrift(ignore_columns=['label_name', 'image_name'])])