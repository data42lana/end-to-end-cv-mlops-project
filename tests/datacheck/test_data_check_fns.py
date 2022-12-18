import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestSamplesMix

# isort: off
from data_checks.check_img_info_and_bbox_csv_file_integrity import (check_that_two_sorted_lists_are_equal,
                                                                    check_that_series_is_less_than_or_equal_to)
from data_checks.check_duplicates_and_two_dataset_similarity import (check_two_datasets, check_bbox_data_for_duplicates, 
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
        assert sorted(check_res['FAILED']) == sorted(['b', 'm'])

    def test_check_that_series_is_less_than_or_equal_to_passed(self):
        s = pd.Series([23, 44, 61])
        check_res = check_that_series_is_less_than_or_equal_to(s, s + 2, '<=', 'Good')
        assert 'PASSED' in check_res
        assert check_res['PASSED'] == 'Good'

    def test_check_that_series_is_less_than_or_equal_to_failed(self):
        s = pd.Series([23, 44, 61])
        check_res = check_that_series_is_less_than_or_equal_to(s, s - 2, '<=')
        assert 'FAILED' in check_res
        assert list(check_res['FAILED']) == list(s.iloc[[0, 1, 2]].index)

class TestSimilarityCheckFns:

    def test_check_two_datasets_passed(self, train_df, val_df):
        cat_features=['Source', 'License']
        ds1, ds2 = [Dataset(df, cat_features=cat_features, index_name='Name') for df in [train_df, val_df]]
        check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0)
        check_suite_res = check_two_datasets(ds1, ds2, 'Test Suite', [check])
        assert check_suite_res.passed(fail_if_check_not_run=True)
    
    def test_check_two_datasets_failed(self, train_df):
        cat_features=['Source', 'License']
        ds = Dataset(train_df, cat_features=cat_features, index_name='Name')
        check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0)
        check_suite_res = check_two_datasets(ds, ds, 'Test Suite', [check])
        assert not check_suite_res.passed(fail_if_check_not_run=True)

    def test_check_bbox_data_for_duplicates_passed(self, bbox_df):
        check_res = check_bbox_data_for_duplicates(bbox_df)
        assert check_res.passed_conditions()

    def test_check_bbox_data_for_duplicates_failed(self, bbox_df):
        bbox_bbox_df = pd.concat([bbox_df, bbox_df], ignore_index=True)
        check_res = check_bbox_data_for_duplicates(bbox_bbox_df)
        assert not check_res.passed_conditions()

    def test_check_two_datasets_similarity_passed(self, train_df, val_df):
        check_suite_res = check_two_datasets_similarity(train_df, val_df)
        assert 'Datasets Size Comparison' == check_suite_res.get_passed_checks()[0].get_header()
        assert 'Train Test Samples Mix' == check_suite_res.get_passed_checks()[1].get_header()

    def test_check_two_datasets_similarity_failed(self, train_df):
        check_suite_res = check_two_datasets_similarity(train_df, train_df)
        assert not check_suite_res.passed(fail_if_check_not_run=True)

    def test_check_train_test_author_group_leakage_passed(self, train_df, val_df):
        check_suite_res = check_train_test_author_group_leakage(train_df, val_df)
        assert check_suite_res.passed(fail_if_check_not_run=True)

    def test_check_train_test_author_group_leakage_failed(self, train_df):
        train_train_df = pd.concat([train_df, train_df], ignore_index=True)
        check_suite_res = check_train_test_author_group_leakage(train_train_df, train_df)
        assert not check_suite_res.passed(fail_if_check_not_run=True)