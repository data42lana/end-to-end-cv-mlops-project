"""This module contains a class and functions to create an image dataloader with(out) transformations."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import albumentations as A # Image Augmentation
import torch # PyTorch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.ops import box_convert
import torchvision.transforms as T

from utils import stratified_group_train_test_split, get_config_yml

# Set partial reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BBOX_FORMATS = {'coco': 'xywh',
                'pascal_voc': 'xyxy',
                'yolo': 'cxcywh'}

def get_image_transforms(box_format):
    """Return an Albumentation object."""
    aug = A.Compose([
                    A.LongestMaxSize(1333, always_apply=True),  
                    A.SmallestMaxSize(800, always_apply=True),
                    A.HorizontalFlip(p=0.6),
                    A.VerticalFlip(p=0.4),
                    A.ColorJitter(0.5, 0.5, 0.5, 0, p=0.7),
                    A.RandomRain(p=0.5),
                    A.OneOrOther(
                        A.Blur(10, p=0.7),
                        A.GaussianBlur((11, 21), p=0.3),
                        p=0.6
                        ),
                    ], 
                    A.BboxParams(format=box_format, label_fields=['labels']),
                    p=0.8)
    return aug

class ImageBBoxDataset(Dataset):
    """A Dataset from CSV to detect objects in images."""
    def __init__(self, csv_file_path, img_dir_path, bbox_path, 
                 img_transforms=None, bbox_transform=None):
        self.img_dir_path = img_dir_path
        self.img_df = pd.read_csv(csv_file_path)
        self.bbox_df = pd.read_csv(bbox_path)
        self.img_transforms = img_transforms
        self.bbox_transform = bbox_transform # (bbox_transform_fn, *bbox_transform_args) 

    def __len__(self):
        return self.img_df.shape[0]

    def __getitem__(self, idx):
        img_name = self.img_df.iloc[idx, 0]
        img_path = self.img_dir_path / img_name
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        bboxes = self.bbox_df.loc[(self.bbox_df.image_name == img_name), 
                                 ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].values
        labels = torch.ones((bboxes.shape[0],), dtype=torch.int64) 

        if self.img_transforms:
            aug = self.img_transforms(image=image, bboxes=bboxes, labels=labels)
            image = aug['image']
            bboxes = aug['bboxes']
                 
        image = T.ToTensor()(image)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float)

        if self.bbox_transform:
            bboxes = self.bbox_transform[0](bboxes, *self.bbox_transform[1:])
             
        target = {'boxes': bboxes,
                  'labels': labels}

        return image, target

def collate_batch(batch):
    """Collate batches in a Dataloader."""
    return tuple(zip(*batch))

def get_train_val_test_dataloaders(batch_size, box_format_before_transform='coco', 
                                   transform_train_imgs=False):
    """Get a data path from configuration file and returns training, validation, 
    and test dataloaders with a box transformation to pascal_voc ('xyxy') format.
    """
    project_path = Path.cwd()
    
    # Get image data paths from a configuration file
    config = get_config_yml(project_path)    
    img_data_paths = config['image_data_paths']
    
    # Set dataset parameters
    img_transforms = get_image_transforms(box_format_before_transform) if transform_train_imgs else None
    bbox_transform = None

    if box_format_before_transform != 'pascal_voc':
        bbox_transform = (box_convert, BBOX_FORMATS[box_format_before_transform], BBOX_FORMATS['pascal_voc'])

    dataset_params = {'img_dir_path': project_path / img_data_paths['images'],
                      'bbox_path': project_path / img_data_paths['bboxes_csv_file'], 
                      'bbox_transform': bbox_transform}  
    # Create datasets
    train_dataset = ImageBBoxDataset(project_path / img_data_paths['train_csv_file'], img_transforms=img_transforms, **dataset_params)
    val_dataset = ImageBBoxDataset(project_path / img_data_paths['train_csv_file'], **dataset_params) 
    test_dataset = ImageBBoxDataset(project_path / img_data_paths['test_csv_file'], **dataset_params)

    # Split data into training and validation sets
    train_ids, val_ids = stratified_group_train_test_split(train_dataset.img_df['Name'], 
                                                           train_dataset.img_df['Number_HSparrows'], 
                                                           train_dataset.img_df['Author'],
                                                           SEED)
    # Create dataloaders
    dl_params = {'batch_size': batch_size,
                 'collate_fn': collate_batch} 
    train_dataloader = DataLoader(Subset(train_dataset, train_ids), shuffle=True, **dl_params)
    val_dataloader = DataLoader(Subset(val_dataset, val_ids), **dl_params)
    test_dataloader = DataLoader(test_dataset, **dl_params)

    return train_dataloader, val_dataloader, test_dataloader