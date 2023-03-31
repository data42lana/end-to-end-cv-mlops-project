"""This module contains a function to load and modify a pre-trained Faster R-CNN
(MobileNet) model.
"""

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def faster_rcnn_mob_model_for_n_classes(num_classes, print_head=False, **load_model_params):
    """Load a pre-trained Faster R-CNN (MobileNet Large) model
    and modify it to classify N classes (true classes + background).

    More information about this model and its parameters can be found at the link:
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py
    """
    # Load a Faster R-CNN model pre-trained on COCO
    faster_rcnn_mob = fasterrcnn_mobilenet_v3_large_fpn(weights='COCO_V1', **load_model_params)

    if print_head:
        print("The Model's Head - Before: \n", faster_rcnn_mob.roi_heads.box_predictor)

    # Get number of input features for the predictor
    in_features_mob = faster_rcnn_mob.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    faster_rcnn_mob.roi_heads.box_predictor = FastRCNNPredictor(in_features_mob,
                                                                num_classes=num_classes)

    if print_head:
        print("The Model's Head - After: \n", faster_rcnn_mob.roi_heads.box_predictor)

    return faster_rcnn_mob
