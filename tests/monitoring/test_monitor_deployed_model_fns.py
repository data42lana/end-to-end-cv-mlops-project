# isort: off
from monitoring.monitor_deployed_model import (get_selected_monitoring_data,
                                               detect_deployed_model_decay_by_scores,
                                               generate_deployed_model_report)
from monitoring.mon_utils import get_monitoring_param_config_yaml, get_number_of_csv_rows


def test_get_selected_monitoring_data(monitoring_data_path):
    model_sel_params = {'reg_model_name': 'model2',
                        'reg_model_version': 2}
    usecols = ['reg_model_name', 'reg_model_version', 'image_width', 'image_height']
    skiprows = [8, 9]
    selected_data = get_selected_monitoring_data(monitoring_data_path, model_sel_params,
                                                 usecols, skiprows)
    assert selected_data.shape[0] == 7
    assert list(selected_data.columns) == usecols
    assert (selected_data['reg_model_name'] == model_sel_params['reg_model_name']).all()
    assert (selected_data['reg_model_version'] == model_sel_params['reg_model_version']).all()


class TestModelDecayDetectionFn:

    def test_detect_deployed_model_decay_by_scores_true(self, monitoring_data_df):
        monitoring_data_df = monitoring_data_df.sort_values('bbox_score', ascending=False)
        model_decay_detected = detect_deployed_model_decay_by_scores(
            monitoring_data_df[:5], monitoring_data_df[5:], 'bbox_score')
        assert model_decay_detected

    def test_detect_deployed_model_decay_by_scores_false(self, monitoring_data_df):
        model_decay_detected = detect_deployed_model_decay_by_scores(
            monitoring_data_df, monitoring_data_df, 'bbox_score')
        assert not model_decay_detected

    def test_detect_deployed_model_decay_by_scores_saved(self, monitoring_data_df, tmp_path):
        check_res_path = tmp_path / 'tcheck_results/model_check_results.html'
        _ = detect_deployed_model_decay_by_scores(
            monitoring_data_df[:5], monitoring_data_df[5:],
            'bbox_score', save_check_results_path=check_res_path)
        assert (check_res_path).exists()


def test_generate_deployed_model_report(monitoring_data_df, tmp_path):
    report_path = tmp_path / 'tmodel_perform_report.html'
    generate_deployed_model_report(monitoring_data_df[:5], monitoring_data_df[5:],
                                   column_mapping=None, color_scheme=None,
                                   save_report_path=report_path)
    assert (report_path).exists()


def test_get_monitoring_param_config_yaml(config_yaml_file, tmp_path):
    yaml_config = get_monitoring_param_config_yaml(tmp_path, config_yaml_file)
    assert yaml_config['image_data_paths']['images'] == 'datas/images'


def test_get_number_of_csv_rows(train_csv_path):
    nrows = get_number_of_csv_rows(train_csv_path, read_column='Number_HSparrows')
    assert nrows == 3
