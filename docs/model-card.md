# Model Card[^*]
This model card provides general information about a model used in this project, its architecture, inputs and outputs, how it was trained, evaluated, etc.
## Model Description
### Model Summary
The model detects house sparrows in photos and is a fine-tuned pre-trained [TorchVision's (PyTorch) implementation](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html#torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn) of the high resolution [Faster R-CNN](https://arxiv.org/abs/1506.01497) model with the [MobileNetV3-Large](https://arxiv.org/abs/1905.02244) [FPN](https://arxiv.org/abs/1612.03144v2) backbone modified to predict two classes (house sparrows and the background).

### Model Details
A model head was modified for two output classes, including the background. The model has two different inputs and outputs depending on a mode it is in: training (in the training mode) and inference (in the evaluation mode). *Details about the inputs and outputs for this section were taken from [`torchvision.models.detection.fasterrcnn_resnet50_fpn`](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn).*

**Type:** Convolutional Neural Network

**Architecture:** *Basic information for this section is taken from [Everything you need to know about TorchVision’s MobileNetV3 implementation (Vasilis Vryniotis and Francisco Massa, May 26, 2021)](https://pytorch.org/blog/torchvision-mobilenet-v3-implementation/).*
- *Detector:* the high resolution [Faster Region-based Convolutional Neural Network (Faster R-CNN)](https://arxiv.org/abs/1506.01497), initialized with weights pre-trained on the [COCO V1 dataset](https://cocodataset.org/#home) with images of 800-1333px.
- *Backbone:* the [MobileNetV3-Large](https://arxiv.org/abs/1905.02244) [Feature Pyramid Network](https://arxiv.org/abs/1612.03144v2) style (FPN-style), initialized with weights pre-trained on [ImageNet 1K V1](https://image-net.org/index.php).

**Parameters:**
The RPN and/or box parameters and their values used to load the model can be found in [these files](#more-information). Information about each of these parameters can be obtained from [the source code of the pre-trained model](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py).

**Input:**
- *Training:*
    - a list of images as torch.Tensor objects with float data type, each of shape [C, H, W] and in range [0.0, 1.0]
    - a list of targets, each of which is a dictionary of:
        - *boxes* as a torch.Tensor object of shape [N, 4] and float data type: the ground-truth bounding box coordinates in the Pascal VOC ([x1, y1, x2, y2]) format
        - *labels* as a torch.Tensor object of the shape [N] and int64 data type: a class label for each ground-truth box.
    Where C, H, and W are the number of image channels, height, and width, respectively, and N is the number of boxes.
- *Inference:* only the list of the image tensors.
The input images can have different sizes.

**Output:**
- *Training:* a dictionary of tensors containing classification and regression losses for both the RPN and the R-CNN (`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`).
- *Inference:* a list of predictions for each input image as a dictionary of:
   - *boxes* as a torch.Tensor object of shape [N, 4] and float data type: predicted bounding box coordinates in the Pascal VOC ([x1, y1, x2, y2]) format
   - *labels* as a torch.Tensor object of the shape [N] and int64 data type: predicted labels for each detection
   - *scores* as a torch.Tensor object of shape [N] and float data type: confidence scores of each detection.
   Where N is the number of the detections.

## Model Training & Evaluation
### Data
**Dataset Details:** The dataset consists of house sparrow photos and their annotation results. The number of house sparrows and their sizes, as well as the sizes of the photos themselves, are various. See the [Dataset Card](./dataset-card.md) for more details about the dataset.

**Preprocessing:**
- *Training:* Augmenting the training dataset by scaling the minimum and maximum sides of the images to certain sizes, flipping them horizontally and vertically, changing their brightness, contrast, and saturation, adding rain effects, and blurring the images. See the [`get_image_transforms`](../src/data/image_dataloader.py) function for more details on the types of image transformations used and their probability of being applied. Converting bounding box data to the Pascal VOC format ([x1, y1, x2, y2]). Scaling the input images to [0.0, 1.0] and converting them, like the boxes and their labels, into torch.Tensor objects.
- *Evaluation/Test:* The same as in the training process, but without the use of image transformation (augmentation).

### Metrics
The performance of the model in this project can be evaluated using precision, recall, F-beta scores calculated for each batch. Only those bounding boxes that had intersection-over-union (IoU) with the ground truth boxes greater than or equal to a IoU threshold were considered correctly detected. Model versions with a value of the metric less than or equal to its initial value were not saved and/or registered.

See the [`object_detection_precision_recall_fbeta_scores`](../src/train/train_inference_fns.py) function for how the metrics are calculated and [these files](#more-information) to find out what metric and thresholds were used.

### Training Details
The pre-trained model with the last 1-6 trainable backbone layer(s) is fine-tuned during training on the collected data. Training details of the pre-trained model itself (`torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn`) can be found in [the training script](https://github.com/pytorch/vision/tree/e35793a1a4000db1f9f99673437c514e24e65451/references/detection#faster-r-cnn-mobilenetv3-large-fpn).

**Type:** fine-tuning

**Parameters:**
A part of training parameters is selected/tuned by hyperparameter optimization if this stage is used in the training process. These and other training parameters can be viewed in [these files](#more-information).

**Algorithm:** The general algorithm for training the model for this project is as follows:
1. Optionally, optimize hyperparameters to search for training parameters that maximize the metric.
2. Train and evaluate the model with default parameters or the best parameters, if found, for more epochs.
3. Register the model and/or save its weights (state dictionary) or checkpoint with the maximum metric value.
4. If necessary, restart the training process from a checkpoint or repeat the algorithm again.

Refer to [DAG](../pipelines/dvc_dag.md) to see how the algorithm is implemented in a training pipeline.

**Technical Specifications:** The model was trained in *Google Colab* using a *GPU*.

## Model Use
### Model Intended Use
The model is intended to be used in a web application via an API to detect and count the number of house sparrows in images uploaded by users.

The intended direct users of the model are users of the web application that was built in this project.

### Malicious and Out-of-Scope Use
Using the model to cause any harm to birds in any way is a malicious use of the model.

The model is trained to detect only live house sparrows (Passer domesticus) older than about 14 days, i.e. already having plumage that distinguishes them from other bird species. The division into subspecies and the evaluation of the model for them were not carried out, so some of them may not be detected.

### Bias & Limitations
The bias of the model was not checked.

The maximum possible number of detected house sparrows in one image is set by the `box_detections_per_img` parameter and cannot be greater than its value.

### How to Use the Model
Run the API server and then the web application provided in this project. Or go to [Web App Demo](https://huggingface.co/spaces/data42lana/how-many-house-sparrows-demo), if available. Upload a photo to it. After a short time, the result will be displayed.

### License
The model may have different licenses depending on the dataset used for training. See [TorchVision LICENSE](https://github.com/pytorch/vision/blob/main/LICENSE) and [Pre-trained Model License](https://github.com/pytorch/vision#pre-trained-model-license) for license information of the pre-trained model.

## More Information
1. Follow the links in this Model Card for more information.
2. Details about the latest version of the model, its performance on test data, and more (if the model is in production) can be found in the [Model Training Pipeline Result Report](../reports/model_report.md).
3. The model loading and training parameters, their values, and other configurations used in the latest training pipeline are contained in the [`params.yaml`](../configs/params.yaml) and [`dvc.lock`](../pipelines/dvc.lock) files. Also the [`best_params.yaml`](../configs/best_params.yaml) contains some training parameter values if hyperparameter optimization has been performed.

[^*]: *Based on [Model Cards for Model Reporting (Margaret Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993v2) and [Hugging Face Hub Model Card Template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md).*
