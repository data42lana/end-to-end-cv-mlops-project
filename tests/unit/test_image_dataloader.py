import random

import albumentations as A
import torch
from torchvision.ops import box_convert

from src.image_dataloader import get_image_transforms, ImageBBoxDataset, create_dataloaders

def test_get_image_transforms():
    img_transform = get_image_transforms('coco')
    assert isinstance(img_transform, A.Compose)

class TestImageBBoxDataset:
    
    def test_imagebboxdataset_is_indexed(self, train_csv_path, imgs_path, bbox_path):
        idx = random.randint(0, 2)
        ds = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path)
        assert torch.is_tensor(ds[idx][0])
        assert torch.is_tensor(ds[idx][1]['boxes'])
        assert torch.is_tensor(ds[idx][1]['labels'])

    def test_imagebboxdataset_with_img_transform(self, train_csv_path, imgs_path, bbox_path):
        idx = random.randint(0, 2)
        ds = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path)
        dstr = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path, 
                                img_transforms=get_image_transforms('coco'))
        assert not torch.equal(ds[idx][0], dstr[idx][0])

    def test_imagebboxdataset_with_bbox_transform(self, train_csv_path, imgs_path, bbox_path):
        idx = random.randint(0, 2)
        ds = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path)
        dstr = ImageBBoxDataset(train_csv_path, imgs_path, bbox_path, 
                                bbox_transform=(box_convert, 'xywh', 'xyxy'))
        xx, yy, w, h = ds[idx][1]['boxes'][0]
        x1, y1, x2, y2 = dstr[idx][1]['boxes'][0]
        assert ds[idx][1]['boxes'].size() ==  dstr[idx][1]['boxes'].size()
        assert [x1, y1, x2, y2] == [xx, yy, xx + w, yy + h]

class TestCreateDataloaders:

    def test_create_one_dataloader(self, imgs_path, train_csv_path, bbox_path):
        dl = create_dataloaders(imgs_path, train_csv_path, bbox_path, 2)
        assert len(dl) == 2
        assert len(dl.dataset) == 3

    def test_create_two_dataloaders(self, imgs_path, train_val_csv_path, bbox_path):
        dl1, dl2 = create_dataloaders(imgs_path, train_val_csv_path, bbox_path, 2,
                                      train_test_split_data=True)
        assert len(dl1) == 2 and len(dl2) == 2
        assert len(dl1.dataset) == 3 and len(dl2.dataset) == 3