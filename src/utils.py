"""This module contains helper functions for model training and inference."""

import io
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as T
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes


def collate_batch(batch):
    """Collate batches in a Dataloader."""
    return tuple(zip(*batch))


def draw_bboxes_on_image(img, bboxes, scores=None, save_img_out_path=None,
                         color='orange', box_width=2, imgsize_in_inches=None):
    """Draw an image with bounding boxes from Tensors and save or show it."""
    if (img.dtype != torch.uint8):
        img = T.functional.convert_image_dtype(img, dtype=torch.uint8)

    img_box = draw_bounding_boxes(img.detach(), boxes=bboxes,
                                  colors=color, width=box_width)
    img = to_pil_image(img_box.detach())

    # Set a fig parameters
    if imgsize_in_inches is None:
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
                    bbox=dict(facecolor=color, alpha=0.5))

    if save_img_out_path:
        if not isinstance(save_img_out_path, io.BytesIO):
            Path(save_img_out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_img_out_path)
        plt.close()
    else:
        plt.show()
    return fig


def draw_and_save_seaborn_plot(seaborn_plot_func=sns.lineplot, *, data=None, x=None, y=None,
                               figsize=(14, 6), x_label="", y_label="", title="",
                               xaxis_grid=False, yaxis_grid=False, x_ticks=None, y_ticks=None,
                               save_file_path=None, **kwargs):
    """Draw a plot using a given seaborn plot function (except 'seaborn.distplot')
    and save it if save_file_path is specified.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax = seaborn_plot_func(data, x=x, y=y, ax=ax, **kwargs)

    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    if x_ticks:
        ax.set_xticks(x_ticks)
    if y_ticks:
        ax.set_yticks(y_ticks)
    ax.xaxis.grid(xaxis_grid)
    ax.yaxis.grid(yaxis_grid)
    ax.set_axisbelow(True)

    if save_file_path:
        save_file_path = Path(save_file_path)
        save_file_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_file_path)
        plt.close()

    return fig


def draw_production_model_metric_history_plots(metric_name, mlclient, registered_model_name,
                                               save_path=None):
    """Draw metric plots for a production stage models and save them
    if save_path is specified.
    """
    production_model_info = mlclient.get_latest_versions(registered_model_name,
                                                         stages=['Production'])
    prod_metric_plots = []
    for prod_info in production_model_info:
        run_id = prod_info.run_id
        metric_history = mlclient.get_metric_history(run_id, metric_name)

        if metric_name == 'f_beta':
            prod_run_params = mlclient.get_run(run_id).data.params
            metric_name += '_{}'.format(prod_run_params.get('eval_beta', ''))

        metric_step_values = collate_batch([(mh.step, mh.value) for mh in metric_history])
        fig = draw_and_save_seaborn_plot(
            x=metric_step_values[0], y=metric_step_values[1],
            figsize=(10, 6), x_label="epochs", y_label=metric_name,
            title=f"{metric_name.capitalize()} Plot",
            save_file_path=save_path / f'plots/metrics/{metric_name}.jpg',
            color='blue' if 'loss' in metric_name else 'orange')
        prod_metric_plots.append(fig)

    return prod_metric_plots


def get_device(use_cuda_config_param):
    """Return torch.device('cpu' or 'cuda') depending on
    the corresponding configuration parameter.
    """
    return torch.device(
        'cuda' if use_cuda_config_param and torch.cuda.is_available() else 'cpu')


def get_latest_registered_pytorch_model(mlclient, registered_model_name, stages=None,
                                        device=torch.device('cpu')):  # noqa: B008
    """Return the latest version of a PyTorch model among registered ones
    with one of given stages if the model exists.
    """
    model_registry_info = mlclient.get_latest_versions(registered_model_name, stages)
    model_versions = [m.version for m in model_registry_info]
    if model_versions:
        model_latest_version = max(model_versions)
        model_uri = 'models:/{}/{}'.format(registered_model_name, model_latest_version)

        latest_pytorch_model = mlflow.pytorch.load_model(model_uri, map_location=device)
        return latest_pytorch_model, model_uri
    else:
        return None


def get_number_of_csv_rows(csv_file_path, read_column=None):
    """Return the number of rows in a CSV file."""
    if read_column is not None:
        read_column = [read_column]
    df = pd.read_csv(csv_file_path, usecols=read_column)
    return df.shape[0]


def get_param_config_yaml(project_path, param_config_file_path='configs/params.yaml'):
    """Get configurations from params.yaml or other yaml files."""
    config_path = project_path / param_config_file_path
    with open(config_path) as conf:
        param_config = yaml.safe_load(conf)
    return param_config


def get_random_img_with_info(csv_file_path, img_dir_path, license_pattern='',
                             random_seed=None):
    """Return a image loaded from a CSV file and selected based on a license pattern
    and return it and its source and author.
    """
    if isinstance(random_seed, int):
        random.seed(random_seed)

    imgs_df = pd.read_csv(csv_file_path,
                          usecols=['Name', 'Author', 'Source', 'License'])
    imgs_df = imgs_df.loc[imgs_df.License.str.contains(license_pattern, regex=False)]

    if not imgs_df.empty:
        image_sample_info = random.choice(imgs_df.to_dict('records'))  # nosec
        image_sample = cv2.cvtColor(cv2.imread(str(img_dir_path / image_sample_info['Name'])),
                                    cv2.COLOR_BGR2RGB)
        return image_sample, image_sample_info
    else:
        return None


def save_model_state(model_to_save, filepath, ckpt_params_dict=None):
    """Save a model state dictionary or a checkpoint."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if (ckpt_params_dict is not None) or isinstance(ckpt_params_dict, dict):
        torch.save({'model_state_dict': model_to_save.state_dict(),
                    **ckpt_params_dict}, filepath)
    else:
        torch.save(model_to_save.state_dict(), filepath)


def stratified_group_train_test_split(data, stratification_basis, groups, random_state=0):
    """Split data in a stratified way into training and test sets,
    taking into account groups, and return the corresponding indices.
    """
    split = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    train_ids, test_ids = next(split.split(X=data, y=stratification_basis, groups=groups))
    return train_ids, test_ids
