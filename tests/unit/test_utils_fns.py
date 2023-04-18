import mlflow
import numpy as np
import pytest
import torch
import torchvision.transforms as T

from src.utils import (collate_batch, draw_and_save_seaborn_plot, draw_bboxes_on_image,
                       draw_production_model_metric_history_plots,
                       get_current_stage_of_registered_model_version, get_device,
                       get_latest_registered_pytorch_model, get_number_of_csv_rows,
                       get_param_config_yaml, get_random_img_with_info,
                       save_model_state, stratified_group_train_test_split)


def test_collate_batch():
    batch = [[11, 22, 33], [44, 55, 66]]
    res_batch = collate_batch(batch)
    assert res_batch == ((11, 44), (22, 55), (33, 66))


@pytest.mark.parametrize('imgidx', [0, 2])
class TestDrawBBoxesOnImage:

    def test_draw_bboxes_on_image_01_tensor(self, imgbboxdataset, imgidx, tmp_path):
        img = imgbboxdataset[imgidx][0]
        bboxes = imgbboxdataset[imgidx][1]['boxes']
        scores = torch.rand(imgbboxdataset[imgidx][1]['labels'].size())
        fpath = tmp_path / 'out.jpg'
        _ = draw_bboxes_on_image(img, bboxes, scores, fpath)
        assert fpath.exists()

    def test_draw_bboxes_on_image_uint8_tensor(self, imgbboxdataset, imgidx, tmp_path):
        img = T.functional.convert_image_dtype(imgbboxdataset[imgidx][0], dtype=torch.uint8)
        bboxes = imgbboxdataset[imgidx][1]['boxes']
        scores = torch.rand(imgbboxdataset[imgidx][1]['labels'].size())
        fpath = tmp_path / 'out.jpg'
        _ = draw_bboxes_on_image(img, bboxes, scores, fpath)
        assert fpath.exists()


def test_draw_and_save_seaborn_plot_is_saved(tmp_path):
    x = [1, 2, 3]
    y = [44, 55, 66]
    fpath = tmp_path / 'plot_123.jpg'
    _ = draw_and_save_seaborn_plot(x=x, y=y, save_file_path=fpath)
    assert fpath.exists()


def test_draw_production_model_metric_history_plots_is_saved(model_registry, tmp_path):
    client, reg_model_name, run_id, _ = model_registry
    client.create_registered_model(reg_model_name)
    client.log_param(run_id, 'eval_beta', 2)
    for i in range(3):
        _ = client.create_model_version(reg_model_name, '', run_id=run_id, await_creation_for=5)
        client.log_metric(run_id, 'f_beta', (i + 1.0)*10, step=i)
    client.transition_model_version_stage(reg_model_name, version='1', stage='Production')
    _ = draw_production_model_metric_history_plots('f_beta', client, reg_model_name, tmp_path)
    assert len([ch for ch in (tmp_path / 'plots/metrics').iterdir()]) == 1


def test_get_current_stage_of_registered_model_version(model_registry):
    client, reg_model_name, run_id, _ = model_registry
    client.create_registered_model(reg_model_name)
    for _ in range(3):
        _ = client.create_model_version(reg_model_name, '', run_id=run_id, await_creation_for=5)
    client.transition_model_version_stage(reg_model_name, version='2', stage='Production')
    current_version_stage = get_current_stage_of_registered_model_version(
        client, reg_model_name, 2)
    assert current_version_stage == 'Production'


def test_get_device():
    device_param = False
    device = get_device(device_param)
    assert device == torch.device('cpu')


def test_get_latest_registered_pytorch_model(model_registry, frcnn_model):
    client, reg_model_name, run_id, exp_id = model_registry
    with mlflow.start_run(run_id, exp_id):
        for _ in range(2):
            mlflow.pytorch.log_model(frcnn_model, reg_model_name,
                                     registered_model_name=reg_model_name,
                                     await_registration_for=5)
    pt_model, model_uri = get_latest_registered_pytorch_model(client, reg_model_name)
    assert isinstance(pt_model, torch.nn.Module)
    assert model_uri == f'models:/{reg_model_name}/2'


def test_get_number_of_csv_rows(train_csv_path):
    nrows = get_number_of_csv_rows(train_csv_path, read_column='Number_HSparrows')
    assert nrows == 3


def test_get_param_config_yaml(config_yaml_file, tmp_path):
    yaml_config = get_param_config_yaml(tmp_path, config_yaml_file)
    assert yaml_config['image_data_paths']['images'] == 'datas/images'


class TestGetRandomImg:

    def test_get_random_img_with_info(self, train_csv_path, imgs_path, train_df):
        img_license_pattern = 'CC0 1.0'
        img_sample, img_sample_info = get_random_img_with_info(train_csv_path, imgs_path,
                                                               license_pattern=img_license_pattern,
                                                               random_seed=0)
        assert isinstance(img_sample, np.ndarray)
        assert img_sample_info['Name'] in train_df.Name.values
        assert sorted(img_sample_info.keys()) == ['Author', 'License', 'Name', 'Source']

    def test_not_get_random_img_with_info(self, train_csv_path, imgs_path):
        img_license_pattern = 'MIT'
        random_img = get_random_img_with_info(train_csv_path, imgs_path,
                                              license_pattern=img_license_pattern,
                                              random_seed=0)
        assert random_img is None


def test_save_model_state(frcnn_model, tmp_path):
    save_model_state(frcnn_model, tmp_path / 'model.pt')
    saved_mst = torch.load(tmp_path / 'model.pt')
    current_mst = frcnn_model.state_dict()
    compared_tensors = [torch.equal(saved_mst[t], current_mst[t]) for t in saved_mst]
    assert sum(compared_tensors) == len(current_mst)


class TestStratifiedGroupTrainTestSplit:

    def test_stratified_group_train_test_split_is_stratified(self, img_info_df):
        train_ids, test_ids = stratified_group_train_test_split(
            img_info_df['Name'], img_info_df['Number_HSparrows'],
            img_info_df['Author'], random_state=0)
        train_num, test_num = [img_info_df.iloc[ids]['Number_HSparrows'].to_list()
                               for ids in [train_ids, test_ids]]
        assert sum([num in train_num for num in [1, 2]]) == 2
        assert sum([num in test_num for num in [1, 2]]) == 2

    def test_stratified_group_train_test_split_contains_different_groups(self, img_info_df):
        train_ids, test_ids = stratified_group_train_test_split(
            img_info_df['Name'], img_info_df['Number_HSparrows'],
            img_info_df['Author'], random_state=0)
        train_auths, test_auths = [
            img_info_df.iloc[ids]['Author'].unique() for ids in [train_ids, test_ids]]
        assert sum([auth in test_auths for auth in train_auths]) == 0
        assert sum([auth in train_auths for auth in test_auths]) == 0
