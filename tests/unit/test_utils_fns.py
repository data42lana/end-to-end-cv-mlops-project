import pytest
import torch
import torchvision.transforms as T

# isort: off
from src.utils import (get_config_yml, get_device, stratified_group_train_test_split, 
                       collate_batch, draw_bboxes_on_image, save_model_state)
                       

def test_get_config_yml():
    yaml_config = get_config_yml()
    assert yaml_config['image_data_paths']['images'] == 'data/raw/images'

def test_get_device():
    device_param = False
    device = get_device(device_param)
    assert device == torch.device('cpu')

class TestStratifiedGroupTrainTestSplit:

    def test_stratified_group_train_test_split_is_stratified(self, img_info_df):
        train_ids, test_ids = stratified_group_train_test_split(
            img_info_df['Name'], img_info_df['Number_HSparrows'], img_info_df['Author'], random_state=0)
        train_num, test_num = [img_info_df.iloc[ids]['Number_HSparrows'].to_list() for ids in [train_ids, test_ids]]
        assert sum([num in train_num for num in [1, 2]]) == 2
        assert sum([num in test_num for num in [1, 2]]) == 2
    
    def test_stratified_group_train_test_split_contains_different_groups(self, img_info_df):
        train_ids, test_ids = stratified_group_train_test_split(
            img_info_df['Name'], img_info_df['Number_HSparrows'], img_info_df['Author'], random_state=0)
        train_auths, test_auths = [img_info_df.iloc[ids]['Author'].unique() for ids in [train_ids, test_ids]]
        assert sum([auth in test_auths for auth in train_auths]) == 0
        assert sum([auth in train_auths for auth in test_auths]) == 0

def test_collate_batch():
    batch = [[11, 22, 33], [44, 55, 66]]
    res_batch = collate_batch(batch)
    assert res_batch == ((11, 44), (22, 55), (33, 66))

@pytest.mark.parametrize('imgidx', [0, 2])
class TestDrawBBoxesOnImage:

    def test_draw_bboxes_on_image_01_tensor(self, imgbboxdataset, imgidx, tmp_path):
        img =imgbboxdataset[imgidx][0]
        bboxes = imgbboxdataset[imgidx][1]['boxes']
        scores = torch.rand(imgbboxdataset[imgidx][1]['labels'].size())
        fpath = tmp_path / 'out.jpg'
        draw_bboxes_on_image(img, bboxes, scores, fpath)
        assert fpath.exists()

    def test_draw_bboxes_on_image_uint8_tensor(self, imgbboxdataset, imgidx, tmp_path):
        img = T.functional.convert_image_dtype(imgbboxdataset[imgidx][0], dtype=torch.uint8)
        bboxes = imgbboxdataset[imgidx][1]['boxes']
        scores = torch.rand(imgbboxdataset[imgidx][1]['labels'].size())
        fpath = tmp_path / 'out.jpg'
        draw_bboxes_on_image(img, bboxes, scores, fpath)
        assert fpath.exists()

def test_save_model_state(frcnn_model, tmp_path):
    save_model_state(frcnn_model, tmp_path / 'model.pt')
    saved_mst = torch.load(tmp_path / 'model.pt')
    current_mst = frcnn_model.state_dict()
    assert sum([torch.equal(saved_mst[t], current_mst[t]) for t in saved_mst]) == len(current_mst)
