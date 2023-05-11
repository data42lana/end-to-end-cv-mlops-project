"""This module contains metrics and functions for a model training-evaluation cycle."""

import gc

import torch
import torchvision.transforms as T
from torchvision.ops import box_iou

from src.utils import draw_bboxes_on_image


@torch.inference_mode()
def object_detection_precision_recall_fbeta_scores(gts, preds, iou_thresh=0.5, beta=1):
    """Calculate precision, recall, and F-beta scores for batches
    of the ground truth and object detection results based on an IoU threshold
    (only 'labels' and 'boxes' from the batches are taken into account).

    During calculation, the boxes are relabeled to find true positives
    and false positives based on a given box IoU threshold.

    Parameters
    ----------
    gts: tuple or list
        The ground truth of label and box values.
    preds: tuple or list
        Predicted label and box values.
    iou_thresh: float, optional
        Minimum IoU between the ground truth bounding boxes and predicted
        bounding boxes to consider them as true positive (default 0.5).
    beta: int, optional
        Beta value to determine the weight of the recall in the F-beta score
        (default 1).

    Return
    ------
        A dictionary containing precision, recall, and F-beta scores.

    Raise
    -----
    ValueError
        When beta or iou_thresh is less than 0.

    Examples
    --------
    >>> ground_truth = {'boxes': torch.tensor([[1.0, 1.0, 4.0, 4.0],
    ...                                        [4.0, 1.0, 7.0, 4.0]]),
    ...                 'labels': torch.tensor([1, 1])}
    >>> precision = {'boxes': torch.tensor([[1.0, 2.0, 4.0, 5.0]]),
    ...              'labels': torch.tensor([1])}
    >>> results = object_detection_precision_recall_fbeta_scores(
    ...    gts=(ground_truth, ground_truth), preds=(prediction, prediction))
    >>> print(results)
    {'precision': 1, 'recall': 0.5, 'f_beta': 0.6666666666666666}
    """
    if (beta or iou_thresh) < 0:
        raise ValueError("beta and iou_thresh must be >=0")

    total_gt_labels = []
    total_correct_pred_labels = []

    for gt, pred in zip(gts, preds):
        total_gt_labels.append(gt['labels'])

        if pred['boxes'].numel() != 0:
            # Box IoU
            gt_pred_box_iou = box_iou(gt['boxes'], pred['boxes'])
            max_ious = torch.max(gt_pred_box_iou, dim=1)

            # Relabel the boxes as true positives (1) and false positives (0)
            # based on a given IoU threshold
            correct_pred_labels = torch.zeros_like(pred['labels'])
            correct_pred_labels[max_ious.indices[max_ious.values >= iou_thresh]] = 1
        else:
            correct_pred_labels = torch.zeros_like(gt['labels'])

        total_correct_pred_labels.append(correct_pred_labels)

    total_correct_pred_labels = torch.cat(total_correct_pred_labels)
    total_gt_labels = torch.cat(total_gt_labels)

    # Precision, recall, and f_beta scores
    tp = sum(total_correct_pred_labels).item()
    recall = tp / total_gt_labels.numel()
    precision = tp / total_correct_pred_labels.numel()
    denom = (beta**2 * precision) + recall
    f_beta = ((1 + beta**2) * (precision * recall)) / denom if denom != 0 else 0

    return {'precision': precision,
            'recall': recall,
            'f_beta': f_beta}


def train_one_epoch(dataloader, model, optimizer, device=torch.device('cpu')):  # noqa: B008
    """Pass a training step in one epoch."""
    accum_dict_losses = {}
    accum_model_loss = 0
    num_batches = len(dataloader)

    # Set a model in training mode
    model.train()

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Compute model batch losses
        batch_dict_losses = model(images, targets)
        batch_model_loss = sum([loss for loss in batch_dict_losses.values()])

        # Accumulate statistics for computing the average losses per epoch
        accum_dict_losses.update({
            k: accum_dict_losses.get(k, 0) + v.item() for k, v in batch_dict_losses.items()
        })
        accum_model_loss += batch_model_loss.item()

        # Optimize the model parameters
        optimizer.zero_grad()
        batch_model_loss.backward()
        optimizer.step()

        # Free up memory
        del images
        del targets
        gc.collect()

        if str(device) == 'cuda':
            torch.cuda.empty_cache()

    # Compute the average losses
    epoch_dict_losses = {k: v / num_batches for k, v in accum_dict_losses.items()}
    epoch_model_loss = accum_model_loss / num_batches

    return {'epoch_dict_losses': epoch_dict_losses,
            'epoch_loss': epoch_model_loss}


@torch.inference_mode()
def eval_one_epoch(dataloader, model, iou_thresh=0.5, beta=1,
                   device=torch.device('cpu')):  # noqa: B008
    """Pass an inference evaluation step in one epoch."""
    accum_model_scores = {}
    results = []
    num_batches = len(dataloader)

    # Set a model in evaluation mode
    model.eval()

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Get prediction results
        outputs = model(images)
        results += outputs

        # Compute model batch statistics
        batch_model_scores = object_detection_precision_recall_fbeta_scores(
            targets, outputs, iou_thresh=iou_thresh, beta=beta)

        # Accumulate statistics for computing the average values per epoch
        accum_model_scores.update({
            k: accum_model_scores.get(k, 0) + v for k, v in batch_model_scores.items()
        })

        # Free up memory
        del images
        del outputs
        gc.collect()

        if str(device) == 'cuda':
            torch.cuda.empty_cache()

    # Compute the average scores
    epoch_model_scores = {k: v / num_batches for k, v in accum_model_scores.items()}

    return {'epoch_scores': epoch_model_scores,
            'results': results}


@torch.inference_mode()
def predict(img, model, device=torch.device('cpu')):  # noqa: B008
    """Return the prediction result containing scores, boxes, and labels."""
    img = T.ToTensor()(img).to(device)
    model.to(device)
    model.eval()
    preds = model([img])[0]
    return preds


@torch.inference_mode()
def predict_image(img, model, show_scores=False, device=torch.device('cpu'),  # noqa: B008
                  save_predict_path=None):
    """Draw an image with bounding boxes (and scores) and return it with
    the number of object detection targets on it.
    """
    preds = predict(img, model, device)
    num_bboxes = len(preds['boxes'])

    scores = None
    if show_scores:
        scores = preds['scores']

    img = T.ToTensor()(img).to(device)
    res_img = draw_bboxes_on_image(img, preds['boxes'], scores,
                                   save_predict_path, imgsize_in_inches=(8, 10))
    return num_bboxes, res_img
