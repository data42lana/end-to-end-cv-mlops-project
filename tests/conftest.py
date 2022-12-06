from pathlib import Path

import pytest
import pandas as pd
import cv2
import yaml
import torch
import torchvision.transforms as T
from torchvision.ops import box_convert

from src.image_dataloader import ImageBBoxDataset

@pytest.fixture(scope='session')
def imgs_path():
    imgs_path = Path('tests/data_samples/sample_imgs').absolute()
    return imgs_path

@pytest.fixture(scope='session')
def img_info_path():
    fpath = Path('tests/data_samples/sample_img_info.csv').absolute()
    return fpath

@pytest.fixture(scope='session')
def bbox_path():
    fpath = Path('tests/data_samples/sample_bboxes.csv').absolute()
    return fpath

@pytest.fixture(scope='session')
def train_csv_path():
    fpath = Path('tests/data_samples/sample_train.csv').absolute()
    return fpath

@pytest.fixture(scope='session')
def test_csv_path():
    fpath = Path('tests/data_samples/sample_test.csv').absolute()
    return fpath

@pytest.fixture(scope='session')
def img_info_df(img_info_path):
    df = pd.read_csv(img_info_path)
    return df

@pytest.fixture(scope='session')
def bbox_df(bbox_path):
    df = pd.read_csv(bbox_path)
    return df

@pytest.fixture(scope='session')
def train_df(train_csv_path):
    df = pd.read_csv(train_csv_path)
    return df

@pytest.fixture(scope='session')
def test_df(test_csv_path):
    df = pd.read_csv(test_csv_path)
    return df

@pytest.fixture(scope='session')
def imgbboxdataset(train_csv_path, imgs_path, bbox_path, request):
    ds = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path, 
                          bbox_transform=(box_convert, 'xywh', 'xyxy'))    
    return ds

# @pytest.fixture(scope='session')
# def config_yml():
#     config = get_config_yml()
#     return config

# @pytest.fixture(scope='class')
# def img(imgs_path, request):
#     img_name = request.node.get_closest_marker('image_name')
#     img = cv2.cvtColor(cv2.imread(str(imgs_path / img_name)), cv2.COLOR_BGR2RGB)
#     return img

# @pytest.fixture(scope='session', params=[1, 2, 7])
# def train_img(train_df, imgs_path, request):
#     df = train_df[train_df['Number_HSparrows'] == request.param]
#     img_name = df['Name'].item()
#     img = cv2.cvtColor(cv2.imread(str(imgs_path / img_name)), cv2.COLOR_BGR2RGB)
#     return {'img_name': img_name, 'img': img}

# @pytest.fixture
# def train_img_tensor(train_img, request):
#     img_tensor_type = request.node.get_closest_marker('img_tensor_type')
#     img_tensor = T.ToTensor()(train_img['img'])

#     if img_tensor_type != 'range_01':
#         img_tensor = T.functional.convert_image_dtype(img_tensor, dtype=torch.uint8)

#     return {'img_name': train_img['img_name'], 'img_tensor': img_tensor}

# @pytest.fixture
# def bbox_tensor(bbox_df):

#     def _bbox_tensor(img_name):
#         bboxes = bbox_df.loc[(bbox_df.image_name == img_name), 
#                             ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].values
#         bbox_t = torch.as_tensor(bboxes, dtype=torch.float)
#         bbox_t = box_convert(bbox_t, 'xywh', 'xyxy')
#         return bbox_t

#     return _bbox_tensor

# @pytest.fixture
# def nn_model():
#     class NN(torch.nn.Module):
#         def __init__(self):
#             super(NN, self).__init__()
#             self.flatten = torch.nn.Flatten()
#             self.linear_relu_stack = torch.nn.Sequential(
#                 torch.nn.Linear(512, 256),
#                 torch.nn.ReLU(),
#                 torch.nn.Linear(256, 5),
#             )

#         def forward(self, x):
#             x = self.flatten(x)
#             out = self.linear_relu_stack(x)
#             return out
    
#     return NN()
    



    

# @pytest.fixture()
# def bbox_bbox_path(bbox_df):
#     bbox_bbox_df = pd.concat([bbox_df, bbox_df], ignore_index=True)
#     fpath = Path('bbox_bbox.csv')
#     bbox_bbox_df.to_csv(fpath, index=False)
#     yield fpath
#     fpath.unlink()

# @pytest.fixture
# def bbox_bbox_path(tmp_path, bbox_df):
#     bbox_bbox_df = pd.concat([bbox_df, bbox_df], ignore_index=True)
#     fpath = tmp_path / 'bbox_bbox.csv'
#     bbox_bbox_df.to_csv(fpath, index=False)
#     return fpath







