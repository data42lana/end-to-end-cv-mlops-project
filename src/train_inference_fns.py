""""This module contains a metric and functions for a model training-evaluation cycle."""

import gc

import torch # PyTorch
from torchvision.ops import box_iou
import torchvision.transforms as T

from utils import draw_bboxes_on_image

@torch.inference_mode()
def precision_recall_fbeta_scores(gts, preds, iou_thresh=0.5, beta=1):
    """Calculate the batch precision, recall, and f_beta scores based on IoU thresholds."""
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

            # Mark box classification results as true and false positive base on a given IoU threshold
            correct_pred_labels = torch.zeros_like(pred['labels'])
            correct_pred_labels[max_ious.indices[max_ious.values >= iou_thresh]] = 1
        else:
            correct_pred_labels = torch.zeros_like(gt['labels'])

        total_correct_pred_labels.append(correct_pred_labels)
    
    total_correct_pred_labels = torch.cat(total_correct_pred_labels)
    total_gt_labels = torch.cat(total_gt_labels)

    # Precision, recall, and f_beta scores'    
    tp = sum(total_correct_pred_labels).item()
    recall = tp / total_gt_labels.numel()
    precision = tp / total_correct_pred_labels.numel()
    denom = (beta**2 * precision) + recall
    f_beta = ((1 + beta**2) * (precision * recall)) / denom if denom !=0 else 0
    
    return {'precision': precision,
            'recall': recall,
            'f_beta': f_beta}

def train_one_epoch(dataloader, model, optimizer, device=torch.device('cpu')):
    """Pass a training step in one epoch."""
    accum_dict_losses = {}
    accum_model_loss = 0
    num_batches = len(dataloader)

    # Set a model to the training mode
    model.train()

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Сompute a model batch losses
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
def eval_one_epoch(dataloader, model, iou_thresh=0.5, beta=1, device=torch.device('cpu')):
    """Pass an inference evaluation step in one epoch."""
    accum_model_scores = {}
    results = []
    num_batches = len(dataloader)
    
    # Set a model to the evaluation mode
    model.eval()
    
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Get prediction results
        outputs = model(images)
        results += outputs

        # Сompute a model batch statistics
        batch_model_scores = precision_recall_fbeta_scores(
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
def predict(img, model, show_scores=False, device=torch.device('cpu')):
    """Draw an image with bounding boxes (and scores) and return
    a number of detection target objects on it.
    """
    img = T.ToTensor()(img).to(device)
    model.to(device)
    model.eval()
    preds = model([img])[0]
    num_bboxes = len(preds['boxes'])

    scores = None
    if show_scores:
        scores = preds['scores']
        
    print(str(num_bboxes) + " house sparrow(s)")
    draw_bboxes_on_image(img, preds['boxes'], scores)    
    return num_bboxes