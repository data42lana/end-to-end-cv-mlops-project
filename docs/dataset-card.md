# Dataset Card[^*]
## Dataset Description
### ***Dataset Summary***
Dataset was created for tasks of detecting and counting house sparrows in photos using a neural network model. It consists of a set of photos of the sparrows in `.jpg` format and two `.csv` files containing annotation results and general information about each photo. The number of house sparrows and their sizes, as well as the sizes of the photos themselves, are various.
### ***Dataset Structure (Instances & Fields)***
* `images` - a directory containing collected photos of house sparrows.
* `image_info.csv` - a file with general information about each photo in the dataset (one row per image):

    - `Name`: image name with extension
    - `Author`: author of the photo
    - `Number_HSparrows`: number of house sparrows in the image
    - `Source`: where was the photo taken from
    - `License`: type of license under which the photo is placed in the source by its author.

* `bboxes/bounding_boxes.csv` - a file containing annotation results in **COCO** format (**xywh**) (one row per bounding box):

    - `label_name`: class name that a bounding box belongs to
    - `bbox_x`: coordinate of the starting point of the box along the x-axis
    - `bbox_y`: coordinate of the starting point of the box along the y-axis
    - `bbox_width`: bounding box width in pixels
    - `bbox_height`: bounding box height in pixels
    - `image_name`: photo name with extension which the box was drawn on
    - `image_width`: image width in pixels
    - `image_height`: image height in pixels.

## Dataset Creation
### ***Data Collection***
The photos of house sparrows (Passer domesticus) were collected manually from [**Flickr**](https://flickr.com) site based on their license type. The search for the necessary photos was carried out by the keyword "house sparrow". When choosing, preference was given to images that differed in size, number and size of the sparrows, and the scene in them. The goal was to create as varied a set as possible. At the same time, a name of a photo, its author, source and type of license and the number of house sparrows in it were entered into the `image_info.csv` file (also manually).
### ***Image Annotation***
Bounding boxes were drawn around each house sparrow manually using [**Make Sense**](https://github.com/SkalskiP/make-sense) tool, and a class label was indicated. The sparrow did not have to be completely visible, the main thing is that the type of bird could be determined from this visible part. At the end of the annotation, results were saved in **COCO** format to the `bounding_boxes.csv` file with automatic addition of a name and size of a corresponding image for each box.
## Data Splitting
### ***Training and Test Dataset Structure (Instances & Fields)***
* `train.csv` - a file containing information about images for training (one row per image):

    - `Name`: image name with extension (taken from `image_info.csv`)
    - `Author`: author of the photo (taken from `image_info.csv`)
    - `Number_HSparrows`: number of house sparrows in the image (taken from `image_info.csv`)
    - `Source`: photo source (taken from `image_info.csv`)
    - `License`: type of photo license (taken from `image_info.csv`)
    - `avg_bbox_width`: average width of the bounding boxes in the image (calculated on data from `bounding_boxes.csv`)
    - `avg_bbox_height`: average height of the bounding boxes in the image (calculated on data from `bounding_boxes.csv`)
    - `image_width`: image width in pixels (taken from `bounding_boxes.csv`)
    - `image_height`: image height in pixels (taken from `bounding_boxes.csv`).

* `test.csv` - a file containing information about images to test. All fields in it are similar to the fields in the `train.csv` file.
### ***EDA of Training Data***
![House Sparrow Distribution Plot](../outputs/plots/eda/train_number_hsparrows_distribution.jpg)
![Author Distribution Plot](../outputs/plots/eda/train_author_distribution.jpg)
![Image Sizes Plot](../outputs/plots/eda/train_img_sizes.jpg)
![Box Sizes Plot](../outputs/plots/eda/train_avg_bbox_sizes.jpg)
### ***Data Split Implementation***
Data splitting into training and test sets is carried out stratified by pseudo-classes, as well as taking into account the belonging of images to a certain group. The `Number_HSparrows` field of the `image_info.csv` file acts as a pseudo class, and `Author` as a group. The process is performed automatically using a specially created function. As a result, the output datasets receive images with approximately the same number of house sparrows, and the sets do not overlap according to the authors. For training purposes, the largest set is always chosen.

Then, information about the average sizes of the bounding boxes and the image sizes is added to the output datasets, and the results are saved to `train.csv` and `test.csv` files. The set of images itself is not divided directly; if necessary, a selection is made from it by their names.

Immediately before training the neural network model, the training dataset is again split in the same way into training and validation sets, but no additional information is entered and the results are not saved to files.
## Additional Information
### ***General Information***
Sample files and images of the dataset are located in `tests/data_samples` directory.

*The complete dataset is uploaded to [**Kaggle**](https://www.kaggle.com/datasets/data42lana/house-sparrow-detection) so that an entire ML pipeline in this project can be reproduce, including with new data.*

To find a specific photo from the dataset in the source ([**Flickr**](https://flickr.com)), we need to go to its author's page and add "/" with the first part of a name of the image (numbers before a "_" sign) to a URL. *For example, we want to find `52089256535_d3b996ae78_w.jpg` photo by `Wildlife Terry`. By going to the author's page, we get the URL: `https://www.flickr.com/photos/wistaston.` By adding `/52089256535` to it, the link to the photo will look like this: `https://www.flickr.com/photos/wistaston/52089256535` (sometimes the author's name and his nickname may be the same).*
### ***Technical Information***
The data is stored on the local file system and versioned by [**DVC**](https://github.com/iterative/dvc). Paths to specific data files and the set of the images are specified in `configs/params.yaml` file(`image_data_paths`), as well as where and how to store new data (`new_image_data_paths`).

The dataset is checked before use in several steps using [**Great Expectations**](https://github.com/great-expectations/great_expectations) (`great_expectations` directory), [**Deepchecks**](https://github.com/deepchecks/deepchecks) tools (`data_checks/check_bbox_duplicates_and_two_dataset_similarity.py`), and additional functions (`data_checks/check_img_info_and_bbox_csv_file_integrity.py`).

The `train.csv` and `test.csv` files are created when `src/data/prepare_data.py` module is run, and the splitting is done using `src\utils.py::stratified_group_train_test_split` function in it.
### ***License Information***
Each photo in the dataset has its own license, which can be found in the `License` field of the `image_info.csv` or  `train.csv` and `test.csv` files. The current license information must be checked on ([**Flickr**](https://flickr.com)) site, and use of the images must abide by the [**Flickr Terms of Use**](https://www.flickr.com/creativecommons/).

[^*]: *Based on [Hugging Face Hub Dataset Card Template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md).*
