import torch

from src.object_detection_model import faster_rcnn_mob_model_for_n_classes
from src.prepare_data import expand_img_df_with_average_values_from_another_img_df

def test_expand_img_df_with_average_values_from_another_img_df(img_info_df, bbox_df, train_df):
    avg_cols = ['bbox_width', 'bbox_height', 'image_width', 'image_height']
    exp_df = expand_img_df_with_average_values_from_another_img_df(img_info_df, bbox_df, img_info_df.Name.iloc[1, 2, 5], 
                                                                   avg_cols, 'Name', 'image_name', avg_cols[:2])
    assert exp_df.eq(train_df)

def test_faster_rcnn_mob_model_for_n_classes():
    model = faster_rcnn_mob_model_for_n_classes(2)
    assert isinstance(model, torch.nn.Module)
    assert model.roi_heads.box_predictor == 2