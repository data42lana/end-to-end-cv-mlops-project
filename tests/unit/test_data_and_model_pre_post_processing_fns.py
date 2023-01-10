import torch

# isort: off
from src.prepare_data import expand_img_df_with_average_values_from_another_img_df
from src.object_detection_model import faster_rcnn_mob_model_for_n_classes
from src.update_model_stages import (update_registered_model_version_stages,
                                     save_production_model_metric_history_plots)


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
    update_registered_model_version_stages(client, reg_model_name)
    assert client.get_model_version(reg_model_name, '2').current_stage == 'Archived'
    assert client.get_model_version(reg_model_name, '3').current_stage == 'Production'
    assert len(client.get_latest_versions(reg_model_name, stages=['Production'])) == 1


def test_save_production_model_metric_history_plots(model_registry, tmp_path):
    client, reg_model_name = model_registry
    save_production_model_metric_history_plots(['metric'], client, reg_model_name, tmp_path)
    assert len([ch for ch in (tmp_path / 'plots').iterdir()]) == 1
