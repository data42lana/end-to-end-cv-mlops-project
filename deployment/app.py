"""This module builds a web application using Streamlit (frontend/UI)."""

import io
from pathlib import Path

import requests
import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.train.train_inference_fns import predict
from src.utils import draw_bboxes_on_image, get_param_config_yaml

FASTAPI_ENDPOINT = 'http://127.0.0.1:8000/detection'  # backend
DEMO_MODEL_PATH = 'models/fine_tuned_model_demo.pt'  # for demo mode


@st.cache_data
def get_box_params(project_path):
    """Get parameters to draw boxes on an image."""
    params = get_param_config_yaml(project_path)
    box_params = params['object_detection_model']['load_parameters']
    return box_params


@st.cache_data
def call_detection_api(byte_image, server_url):
    """Get object detection model inferences using an API (backend)."""
    response = requests.post(server_url, files={'input_image': byte_image})
    return response.json()


@st.cache_data
def detect_objects_demo(byte_image, project_path):
    """Get a object detection model inference without an API (backend)."""
    img = Image.open(byte_image).convert('RGB')
    device = torch.device('cpu')
    model = torch.load(project_path / DEMO_MODEL_PATH,
                       map_location=device)
    result = predict(img, model)
    result = {k: v.tolist() for k, v in result.items()}
    return result


def main(project_path, demo=False):
    """Build a web app with a connection to a given API endpoint (backend),
    or using a demo model if demo is set to True (demo mode).
    """
    MODEL_BOX_PARAMS = get_box_params(project_path)
    STATIC_IMG = {'name': 'detected_36485871561.png',
                  'author': 'Wildlife Terry',
                  'caption': 'Hungry Sparrows',
                  'link': 'https://www.flickr.com/photos/wistaston/36485871561',
                  'source': 'Flickr',
                  'source_link': 'https://flickr.com',
                  'license': 'CC0 1.0'}

    # Add titles, description, and a static image
    add_to_title = "Demo" if demo else ""
    st.set_page_config(page_title=f"HouseSparrowDetector{add_to_title}")
    st.title(f":orange[How many House Sparrows?] {add_to_title}")
    st.write("##### *Detect and count house sparrows in a photo*")
    st.write("![{0}](app/static/{1})".format(STATIC_IMG['caption'], STATIC_IMG['name']))
    st.caption("Photo by [{0}]({1}) on [{2}]({3}). License: {4}. "
               "*Photo modified: cropped, boxes and scores drawn*".format(
                   STATIC_IMG['author'], STATIC_IMG['link'], STATIC_IMG['source'],
                   STATIC_IMG['source_link'], STATIC_IMG['license']))

    # Add an image upload widget
    uploaded_image = st.file_uploader("Choose a photo", type=['png', 'jpg', 'jpeg'])
    st.caption(
        "The maximum possible number of house sparrows to be detected is {}".format(
            MODEL_BOX_PARAMS['box_detections_per_img']))

    # Add widgets to a sidebar with box display options
    with st.sidebar:
        st.write("### **Box Display Settings**")
        min_sc_thresh = MODEL_BOX_PARAMS['box_score_thresh']
        bbox_sc_thresh = st.sidebar.slider(
            "Min Box Score", min_sc_thresh, 1.0, min_sc_thresh,
            help="Show only boxes with a score greater than a given threshold")
        show_scores = st.checkbox("Show Scores")
        bbox_color = st.sidebar.color_picker("Box Color", "#FFA500")

    if uploaded_image is not None:
        # Get model inference
        if demo:
            detect_res = detect_objects_demo(uploaded_image, project_path)
        else:
            detect_res = call_detection_api(uploaded_image.getvalue(),
                                            FASTAPI_ENDPOINT)

        # Display original and with detections images
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Original")
            original_image = Image.open(uploaded_image)
            st.image(original_image, use_column_width=True)

        with col2:
            st.write("#### Detected")
            select_sc_mask = torch.tensor(detect_res['scores']) >= bbox_sc_thresh
            boxes = torch.tensor(detect_res['boxes'])[select_sc_mask]
            scores = torch.tensor(detect_res['scores'])[select_sc_mask] if show_scores else None
            buf = io.BytesIO()
            _ = draw_bboxes_on_image(TF.pil_to_tensor(original_image), boxes,
                                     scores, color=bbox_color, save_img_out_path=buf)
            buf.seek(0)
            st.image(Image.open(buf), use_column_width=True)

        if boxes.numel() == 0:
            st.warning(
                "No house sparrows were detected in the photo. "
                "Try uploading another photo or changing the box display settings",
                icon="😕")
        else:
            st.success("{} house sparrow(s) was(were) detected".format(len(boxes)), icon="😃")


if __name__ == '__main__':
    project_path = Path.cwd()
    # Use the demo model if demo is set to True (demo mode)
    # or the fastapi backend otherwise
    main(project_path, demo=True)
