import mlflow
import torch

from src.prepare_data import expand_img_df_with_average_values_from_another_img_df
from src.object_detection_model import faster_rcnn_mob_model_for_n_classes
from src.update_model_stages import update_registered_model_version_stages

def test_expand_img_df_with_average_values_from_another_img_df(img_info_df, bbox_df, train_df):
    avg_cols = ['bbox_width', 'bbox_height', 'image_width', 'image_height']
    exp_df = expand_img_df_with_average_values_from_another_img_df(img_info_df, bbox_df, img_info_df.Name.iloc[1, 2, 5], 
                                                                   avg_cols, 'Name', 'image_name', avg_cols[:2])
    assert exp_df.eq(train_df)

def test_faster_rcnn_mob_model_for_n_classes():
    model = faster_rcnn_mob_model_for_n_classes(2)
    assert isinstance(model, torch.nn.Module)
    assert model.roi_heads.box_predictor == 2

def test_update_registered_model_version_stages(mltracking): # tmp_path
    reg_model_name = 'test_frcnn_reg'
    # mlflow.set_tracking_uri(f'sqlite:////{tmp_path}/mlruns.db')
    client = mlflow.MlflowClient()
    client.create_registered_model(reg_model_name)
    for i in range(3):
        _ = client.create_model_version(reg_model_name, '')
    client.transition_model_version_stage(reg_model_name, version='2', stage='Production')
    
    update_registered_model_version_stages(client, reg_model_name)
    
    assert client.get_model_version(reg_model_name, '2').current_stage == 'Archived'
    assert client.get_model_version(reg_model_name, '3').current_stage == 'Production'

