"""This module contains helper functions."""

from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
import torch # PyTorch
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

def get_device(config_param):
    """Returns torch.device('cpu' or 'cuda') depending on the corresponding configuration parameter."""
    return torch.device('cuda' if config_param and torch.cuda.is_available() else 'cpu')

def stratified_group_train_test_split(data, stratification_basis, groups, random_state=0):
    """Stratified splits data into training and test sets,
    taking into account groups, and returns the corresponding indices.
    """
    split = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    train_ids, test_ids = next(split.split(X=data, y=stratification_basis, groups=groups))
    return train_ids, test_ids

def draw_bboxes_on_image(img, bboxes, scores=None):
    """Draws an image with bounding boxes from Tensors."""
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
            ax.text(x, y, text_sc , fontsize=12, 
                    bbox=dict(facecolor='orange', alpha=0.5))            
    plt.show()

def save_model_state(model_to_save, filepath, ckpt_params_dict=None):
    """Saves a model state dictionary or a checkpoint."""
    if (ckpt_params_dict is not None) or isinstance(ckpt_params_dict, dict):
        torch.save({'model_state_dict': model_to_save.state_dict(),
                    **ckpt_params_dict}, filepath)
    else:
        torch.save(model_to_save.state_dict(), filepath)