from pathlib import Path

import cv2
import optuna
import pandas as pd
import pytest
import yaml
from torchvision.ops import box_convert

from src.image_dataloader import ImageBBoxDataset, create_dataloaders
from src.object_detection_model import faster_rcnn_mob_model_for_n_classes


@pytest.fixture(scope='package')
def imgs_path():
    imgs_path = Path('tests/data_samples/sample_imgs').absolute()
    return imgs_path

@pytest.fixture(scope='package')
def img_info_df():
    fpath = Path('tests/data_samples/sample_img_info.csv').absolute()
    df = pd.read_csv(fpath)
    return df

@pytest.fixture(scope='package')
def dataloader(imgs_path, train_csv_path, bbox_path):
    dl = create_dataloaders(imgs_path, train_csv_path, bbox_path, 2)
    return dl

@pytest.fixture(scope='package')
def frcnn_model():
    model = faster_rcnn_mob_model_for_n_classes(2)
    return model

@pytest.fixture(scope='module')
def hp_conf():
    hp_conf = yaml.safe_load("""
    metric: f_beta
    epochs: 3
    hyperparameters: 
        optimizers: 
            SGD: 
                lr: 
                - low: 0.0001
                  high: 0.01
                - float
            Adam: 
                lr: 
                - low: 0.0001
                  high: 0.01
                - float
        lr_schedulers: 
            StepLR:
                step_size: 
                    - low: 1
                      high: 3
                    - int
            None: null
    """)
    return hp_conf

@pytest.fixture(scope='module')
def simple_study():    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(0))
    study.add_trial(optuna.trial.create_trial(
        params={'optimizer': 'SGD',
                'lr': 0.002,
                'lr_scheduler': 'StepLR',
                'step_size': 2},
        distributions={'optimizer': optuna.distributions.CategoricalDistribution(['SGD', 'Adam']),
                       'lr': optuna.distributions.FloatDistribution(0.0001, 0.01),
                       'lr_scheduler': optuna.distributions.CategoricalDistribution(['StepLR', 'None']),
                       'step_size': optuna.distributions.IntDistribution(1, 3)},
        value=0.52))
    study.add_trial(optuna.trial.create_trial(
        params={'optimizer': 'Adam',
                'lr': 0.001,
                'lr_scheduler': 'None'},
        distributions={'optimizer': optuna.distributions.CategoricalDistribution(['SGD', 'Adam']),
                       'lr': optuna.distributions.FloatDistribution(0.0001, 0.01),
                       'lr_scheduler': optuna.distributions.CategoricalDistribution(['StepLR', 'None'])},
        value=0.79))
    return study

@pytest.fixture(scope='class')
def imgbboxdataset(train_csv_path, imgs_path, bbox_path):
    ds = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path, 
                          bbox_transform=(box_convert, 'xywh', 'xyxy'))    
    return ds

@pytest.fixture
def train_val_csv_path():
    fpath = Path('tests/data_samples/sample_train_val.csv').absolute()
    return fpath

@pytest.fixture
def img(imgs_path, train_df):
    img_path = imgs_path / train_df.Name.iloc[0]
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    return img
