import pytest
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestSamplesMix

from data_checks.check_img_info_and_bbox_csv_file_integrity import (check_that_two_sorted_lists_are_equal,
                                                                    check_that_series_is_less_than_or_equal_to)
from data_checks.check_duplicates_and_two_dataset_similarity import (check_two_datasets, check_bbox_csv_file_for_duplicates,
                                                                     check_two_datasets_similarity, 
                                                                     check_train_test_author_group_leakage) 

class TestIdentityCheckFns:
    
    def test_check_that_two_sorted_lists_are_equal_passed(self):
        check_res = check_that_two_sorted_lists_are_equal(['b', 'c', 'a'], ['a', 'c', 'b'], 'Good')
        assert check_res.get('PASSED')
        assert check_res['PASSED'] == 'Good'

    def test_check_that_two_sorted_lists_are_equal_found_duplicates(self):
        check_res = check_that_two_sorted_lists_are_equal(['b', 'c', 'a'], ['b', 'c', 'b'])
        assert check_res.get('WARNING: Duplicates!')
        assert check_res['WARNING: Duplicates!'] == 1

    def test_check_that_two_sorted_lists_are_equal_failed(self):
        check_res = check_that_two_sorted_lists_are_equal(['b', 'c', 'a'], ['m', 'c', 'a'])
        assert check_res.get('FAILED')
        assert check_res['FAILED'] == ['b', 'm']

    def test_check_that_series_is_less_than_or_equal_to_passed(self):
        s = pd.Series([23, 44, 61])
        check_res = check_that_series_is_less_than_or_equal_to(s, s + 2, '<=', 'Good')
        assert check_res.get('PASSED')
        assert check_res['PASSED'] == 'Good'

    def test_check_that_series_is_less_than_or_equal_to_failed(self):
        s = pd.Series([23, 44, 61])
        check_res = check_that_series_is_less_than_or_equal_to(s, s - 2, '<=')
        assert check_res.get('FAILED')
        assert check_res['FAILED'] == s.iloc[[0, 1, 2]].index
        # pd.testing.assert_frame_equal(check_res['FAILED'], s.iloc[[0, 1, 2]].index)

class TestSimilarityCheckFns:

    def test_check_two_datasets_passed(self, train_df, test_df):
        cat_features=['Source', 'License']
        ds1, ds2 = [Dataset(df, cat_features=cat_features, index_name='Name') for df in [train_df, test_df]]
        check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0)
        check_res = check_two_datasets(ds1, ds2, 'Test Suite', [check])
        assert check_res.passed_conditions()
    
    def test_check_two_datasets_failed(self, train_df):
        cat_features=['Source', 'License']
        ds = Dataset(train_df, cat_features=cat_features, index_name='Name')
        check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0)
        check_suite_res = check_two_datasets(ds, ds, 'Test Suite', [check])
        assert not check_suite_res.passed()

    def test_check_bbox_csv_file_for_duplicates_passed(self, bbox_path):
        check_res = check_bbox_csv_file_for_duplicates(bbox_path)
        assert check_res.passed_conditions()

    def test_check_bbox_csv_file_for_duplicates_failed(self, bbox_bbox_path):
        check_res = check_bbox_csv_file_for_duplicates(bbox_bbox_path)
        assert not check_res.passed_conditions()

    def test_check_two_datasets_similarity_passed(self, train_csv_path, test_csv_path):
        check_suite_res = check_two_datasets_similarity(train_csv_path, test_csv_path)
        assert check_suite_res.passed()

    def test_check_two_datasets_similarity_failed(self, train_csv_path):
        check_suite_res = check_two_datasets_similarity(train_csv_path, train_csv_path)
        assert not check_suite_res.passed()

    def test_check_train_test_author_group_leakage_passed(self, train_csv_path, test_csv_path):
        check_suite_res = check_train_test_author_group_leakage(train_csv_path, test_csv_path)
        assert check_suite_res.passed()

    def test_check_train_test_author_group_leakage_failed(self, train_csv_path):
        check_suite_res = check_train_test_author_group_leakage(train_csv_path, train_csv_path)
        assert not check_suite_res.passed()