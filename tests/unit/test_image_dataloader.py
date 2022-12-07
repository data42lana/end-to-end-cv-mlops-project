import albumentations as A
import torch
from torchvision.ops import box_convert

from src.image_dataloader import get_image_transforms, ImageBBoxDataset, create_dataloaders

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

def test_create_dataloader(imgs_path, train_val_path, bbox_path):
    dl = create_dataloaders(imgs_path, train_val_path, bbox_path, 2)
    assert len(dl) == 3
    assert len(dl.dataset) == 6

def test_create_two_dataloaders(imgs_path, train_val_path, bbox_path):
    dl1, dl2 = create_dataloaders(imgs_path, train_val_path, bbox_path, 2,
                                  train_test_split_data=True)
    assert len(dl1) == 2 and len(dl2) == 2
    assert len(dl1.dataset) == 3 and len(dl2.dataset) == 3