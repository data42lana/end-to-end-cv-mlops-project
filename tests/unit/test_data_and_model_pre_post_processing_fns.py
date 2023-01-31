import shutil

import pandas as pd
import torch

# isort: off
from src.data.update_raw_data import update_dir_or_csv_files
from src.data.prepare_data import expand_img_df_with_average_values_from_another_img_df
from src.model.object_detection_model import faster_rcnn_mob_model_for_n_classes
from src.model.update_model_stages import (update_registered_model_version_stages,
                                           production_model_metric_history_plot)


class TestUpdateData:

    def test_update_dir(self, imgs_path, tmp_path):
        img_names = [ch for ch in imgs_path.iterdir()]
        copy_imgs_path = tmp_path / 'sample_imgs'
        copy_imgs_path.mkdir()
        update_dir_or_csv_files(imgs_path, copy_imgs_path)
        assert len([ch for ch in copy_imgs_path.iterdir()]) == len(img_names)

    def test_update_csv_file(self, train_csv_path, val_csv_path, train_val_df, tmp_path):
        file_to_update = tmp_path / 'file_to_update.csv'
        file_to_update.touch()
        _ = shutil.copy2(train_csv_path, file_to_update)
        update_dir_or_csv_files(val_csv_path, file_to_update)
        updated_df = pd.read_csv(file_to_update).sort_values('Name', ignore_index=True)
        assert updated_df.equals(train_val_df.sort_values('Name', ignore_index=True))

    def test_not_update_if_not_dir_or_csv_files(self, tmp_path):
        file_to_update = tmp_path / 'file_to_update.txt'
        file_to_update.write_text('Hello')
        new_file = tmp_path / 'new_file.txt'
        new_file.write_text('World')
        update_dir_or_csv_files(new_file, file_to_update)
        assert file_to_update.read_text() == 'Hello'


def test_expand_img_df_with_average_values_from_another_img_df(img_info_df, bbox_df, train_df):
    avg_cols = ['bbox_width', 'bbox_height', 'image_width', 'image_height']
    exp_df = expand_img_df_with_average_values_from_another_img_df(
        img_info_df, bbox_df, img_info_df.Name.iloc[[1, 2, 5]],
        avg_cols, 'Name', 'image_name', avg_cols[:2])
    assert exp_df.equals(train_df)


def test_faster_rcnn_mob_model_for_n_classes():
    model = faster_rcnn_mob_model_for_n_classes(2)
    assert isinstance(model, torch.nn.Module)
    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    assert model.roi_heads.box_predictor.bbox_pred.out_features == 8


def test_update_registered_model_version_stages(model_registry):
    client, reg_model_name = model_registry
    _ = update_registered_model_version_stages(client, reg_model_name)
    assert client.get_model_version(reg_model_name, '2').current_stage == 'Archived'
    assert client.get_model_version(reg_model_name, '3').current_stage == 'Production'
    assert len(client.get_latest_versions(reg_model_name, stages=['Production'])) == 1


def test_production_model_metric_history_plot_is_saved(model_registry, tmp_path):
    client, reg_model_name = model_registry
    _ = production_model_metric_history_plot('f_beta', client, reg_model_name, tmp_path)
    assert len([ch for ch in (tmp_path / 'plots').iterdir()]) == 1
