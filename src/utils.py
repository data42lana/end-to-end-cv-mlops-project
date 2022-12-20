"""This module contains helper functions for model training."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes


def get_config_yml():
    """Get configurations from a yaml file."""
    config_path = Path.cwd() / 'configs/config.yaml'
    with open(config_path) as conf:
        config = yaml.safe_load(conf)
    return config


def get_device(use_cuda_config_param):
    """Return torch.device('cpu' or 'cuda') depending on
    the corresponding configuration parameter.
    """
    return torch.device(
        'cuda' if use_cuda_config_param and torch.cuda.is_available() else 'cpu')


def stratified_group_train_test_split(data, stratification_basis, groups, random_state=0):
    """Split data in a stratified way into training and test sets,
    taking into account groups, and return the corresponding indices.
    """
    split = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    train_ids, test_ids = next(split.split(X=data, y=stratification_basis, groups=groups))
    return train_ids, test_ids


def collate_batch(batch):
    """Collate batches in a Dataloader."""
    return tuple(zip(*batch))


def draw_bboxes_on_image(img, bboxes, scores=None, save_img_out_path=None):
    """Draw or save an image with bounding boxes from Tensors."""
    if (img.dtype != torch.uint8):
        img = T.functional.convert_image_dtype(img, dtype=torch.uint8)

    img_box = draw_bounding_boxes(img.detach(), boxes=bboxes, colors='orange', width=2)
    img = to_pil_image(img_box.detach())
    plt.figure(figsize=(8, 10))
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()

    if scores is not None:
        for bb, sc in zip(bboxes, scores):
            x, y = bb.tolist()[:2]
            text_sc = f"{sc:0.2f}"
            ax.text(x, y, text_sc, fontsize=12,
                    bbox=dict(facecolor='orange', alpha=0.5))

    if save_img_out_path:
        Path(save_img_out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_img_out_path)
        plt.close()
    else:
        plt.show()


def save_model_state(model_to_save, filepath, ckpt_params_dict=None):
    """Save a model state dictionary or a checkpoint."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if (ckpt_params_dict is not None) or isinstance(ckpt_params_dict, dict):
        torch.save({'model_state_dict': model_to_save.state_dict(),
                    **ckpt_params_dict}, filepath)
    else:
        torch.save(model_to_save.state_dict(), filepath)
