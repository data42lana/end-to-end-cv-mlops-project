"""This module builds a web application demo using Streamlit."""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import draw_bounding_boxes

# Set a model to use and its display parameters
DEMO_MODEL = 'fine_tuned_faster_rcnn_mob_large_demo.pt'
MAX_NUM_BOXES_ON_IMG = 120
MIN_BOX_SCORE_THRESHOLD = 0.5


@st.cache_resource(ttl=3600, max_entries=20)
@torch.inference_mode()
def detect_objects_demo(byte_image, project_path):
    """Get an object detection model inference."""
    device = torch.device('cpu')
    img = T.ToTensor()(Image.open(byte_image).convert('RGB')).to(device)
    model = torch.load(project_path / DEMO_MODEL,
                       map_location=device)
    model.eval()
    result = model([img])[0]
    result = {k: v.tolist() for k, v in result.items()}
    return result


def draw_bboxes_on_image_and_save_it_in_memory(img, bboxes, buffered_io_object_to_save_img,
                                               scores=None, box_color='orange'):
    """Draw an image with bounding boxes from Tensors and save the result
    in memory.
    """
    if (img.dtype != torch.uint8):
        img = T.functional.convert_image_dtype(img, dtype=torch.uint8)

    img_box = draw_bounding_boxes(img.detach(), boxes=bboxes,
                                  colors=box_color, width=2)
    img = TF.to_pil_image(img_box.detach())

    # Set figure parameters
    imgsize_in_inches = tuple(map(lambda x: x / 100, img.size))
    fig = plt.figure(figsize=imgsize_in_inches, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)
    ax.margins(0)
    ax.axis('off')

    if scores is not None:
        for bb, sc in zip(bboxes, scores):
            x, y = bb.tolist()[:2]
            text_sc = f"{sc:0.2f}"
            ax.text(x, y, text_sc, fontsize=12,
                    bbox=dict(facecolor='orange', alpha=0.5))

    plt.savefig(buffered_io_object_to_save_img)
    plt.close()
    return fig


def main(project_path):
    """Build a web app demo using a demo model."""
    # Add a title
    st.title(":orange[How many House Sparrows?] Demo")
    st.write("##### *Detect and count house sparrows in a photo*")

    # Add an image upload widget
    uploaded_image = st.file_uploader("Choose a photo", type=['png', 'jpg', 'jpeg'])
    st.caption(
        "The maximum possible number of house sparrows to be detected is {}".format(
            MAX_NUM_BOXES_ON_IMG))

    # Add widgets to a sidebar with box display options
    with st.sidebar:
        st.write("### **Box Display Settings**")
        min_sc_thresh = MIN_BOX_SCORE_THRESHOLD
        bbox_sc_thresh = st.sidebar.slider(
            "Min Box Score", min_sc_thresh, 1.0, min_sc_thresh,
            help="Show only boxes with scores greater than the selected threshold")
        show_scores = st.checkbox("Show Scores")
        bbox_color = st.sidebar.color_picker("Box Color", "#FFA500")

    if uploaded_image is not None:
        # Get a model inference
        detect_res = detect_objects_demo(uploaded_image, project_path)

        # Display an original image and the image with detection results
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
            _ = draw_bboxes_on_image_and_save_it_in_memory(
                TF.pil_to_tensor(original_image), boxes, scores=scores,
                box_color=bbox_color, buffered_io_object_to_save_img=buf)
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
    main(project_path)
