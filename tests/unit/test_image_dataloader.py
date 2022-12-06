import pytest
import albumentations as A
import torch
from torchvision.ops import box_convert

from src.image_dataloader import (get_image_transforms, ImageBBoxDataset, 
                                  collate_batch, create_dataloaders)

def test_get_image_transforms():
    img_transform = get_image_transforms('coco')
    assert isinstance(img_transform, A.Compose)

class TestImageBBoxDataset:
    
    def test_imagebboxdataset_is_indexed(train_csv_path, imgs_path, bbox_path):
        ds = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path)
        assert isinstance(ds[0][0], torch.Tensor)
        assert isinstance(ds[0][1], dict)

    def test_imagebboxdataset_with_img_transform(train_csv_path, imgs_path, bbox_path):
        ds = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path)
        dstr = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path, 
                                img_transforms=get_image_transforms('coco'))
        assert torch.equal(ds[0][0], dstr[0][0])

    def test_imagebboxdataset_with_bbox_transform(train_csv_path, imgs_path, bbox_path):
        ds = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path)
        dstr = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path, 
                                bbox_transform=(box_convert, 'xywh', 'xyxy'))
        assert torch.equal(ds[0][1]['bboxes'], dstr[0][1]['bboxes'])
    
    def test_imagebboxdataset_with_img_and_bbox_transform(train_csv_path, imgs_path, bbox_path):
        ds = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path)
        dstr = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path,
                                img_transforms=get_image_transforms('coco'), 
                                bbox_transform=(box_convert, 'xywh', 'xyxy'))
        assert torch.equal(ds[0][1]['bboxes'], dstr[0][1]['bboxes'])

def test_collate_batch():
    batch = [[11, 22, 33], [44, 55, 66]]
    res_batch = collate_batch([])
    assert res_batch == ([11, 44], [22, 55], [33, 66])

# def test_create_dataloaders():
#     train_dl, val_dl, test_dl = create_dataloaders(2, transform_train_imgs=True)

#     img_dir_path, csv_file_path, bboxes_path, batch_size, 
#                        box_format_before_transform='coco', train_test_split_data=False, 
#                        transform_train_imgs=False