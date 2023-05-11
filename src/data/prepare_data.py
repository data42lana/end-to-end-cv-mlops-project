"""This module creates training and test datasets from raw data
and saves EDA plots for them.
"""

import logging
from pathlib import Path

import pandas as pd
import seaborn as sns

from src.utils import (draw_and_save_seaborn_plot, get_param_config_yaml,
                       stratified_group_train_test_split)

logging.basicConfig(level=logging.INFO, filename='pipe.log',
                    format="%(asctime)s -- [%(levelname)s]: %(message)s")

# Set reproducibility
SEED = 0


def expand_img_df_with_average_values_from_another_img_df(df1, df2,
                                                          selected_images,
                                                          df2_columns_to_calculate_averages,
                                                          df1_image_name_column,
                                                          df2_image_name_column,
                                                          df2_columns_to_rename_in_new_df):
    """Expand a pd.DataFrame with selected images with columns from another pd.DataFrame
    with averages calculated for each group of these images.

    Parameters
    ----------
    df1: pd.DataFrame
        pd.DataFrame object to expand.
    df2: pd.DataFrame
        pd.DataFrame object to calculate averages.
    selected_images: list
        The images for which average values from df2 will be calculated.
    df2_columns_to_calculate_averages: list
        df2 columns to calculate average values.
    df1_image_name_column: str
        df1 column with image names to merge.
    df2_image_name_column: str
        df2 column with image names to merge.
    df2_columns_to_rename_in_new_df: list
        df2 columns with averages to be renamed.

    Return
    ------
        A new expanded pd.DataFrame object.
    """
    rename_columns = {df2_image_name_column: df1_image_name_column}

    if df2_columns_to_rename_in_new_df:
        rename_columns.update({col: 'avg_' + col for col in df2_columns_to_rename_in_new_df})

    avg_df = (df2.loc[df2[df2_image_name_column].isin(selected_images),
                      df2_columns_to_calculate_averages + [df2_image_name_column]]
                 .groupby(df2_image_name_column)
                 .agg('mean')
                 .round()
                 .reset_index()
                 .rename(columns=rename_columns))

    new_expanded_df = (df1.loc[df1[df1_image_name_column].isin(selected_images)]
                          .merge(avg_df, on=df1_image_name_column, how='left'))
    return new_expanded_df


def main(project_path, param_config, save_eda_plots=False):
    """Split data into training and test sets, save them to CSV files,
    and create and return EDA plots for them.
    """
    # Get image data paths from the configurations
    img_data_paths = param_config['image_data_paths']

    # Split data into training and test sets
    img_info_df, img_bbox_df = [
        pd.read_csv(project_path / img_data_paths[csv_data_file])
        for csv_data_file in ['info_csv_file', 'bboxes_csv_file']]
    train_ids, test_ids = stratified_group_train_test_split(img_info_df['Name'],
                                                            img_info_df['Number_HSparrows'],
                                                            img_info_df['Author'],
                                                            SEED)
    # The training set must be larger than the test one
    if train_ids.size < test_ids.size:
        train_ids, test_ids = test_ids, train_ids

    # Create and save training and test CSV files
    train_test_dfs = []
    for ids, fpath in zip((train_ids, test_ids), ('train_csv_file', 'test_csv_file')):
        fpath = project_path / img_data_paths[fpath]
        fpath.parent.mkdir(exist_ok=True)

        sel_imgs = img_info_df.Name.iloc[ids]
        cols_to_calculate_avg = ['bbox_width', 'bbox_height', 'image_width', 'image_height']
        expanded_df = expand_img_df_with_average_values_from_another_img_df(
            img_info_df, img_bbox_df, sel_imgs, cols_to_calculate_avg,
            'Name', 'image_name', cols_to_calculate_avg[:2])
        train_test_dfs.append(expanded_df)
        expanded_df.to_csv(fpath, index=False)
        logging.info("Train and test CSV files are saved.")

    # Create and save EDA plots for training data
    train_df = train_test_dfs[0]
    # sns.set_theme(style='whitegrid')
    eda_plots = []
    save_path = None

    if save_eda_plots:
        save_path = (project_path.joinpath(
            param_config['model_training_inference_conf']['save_model_output_dir'],
            'plots/eda'))

    # Draw distribution plots
    distribution_params = {
        'Number_HSparrows': {
            'x_label': "Number of house sparrows",
            'y_label': "Number of images",
            'title': "House Sparrow Distribution by Images (Train, {} images)".format(
                train_df.shape[0])},
        'Author': {
            'x_label': "Number of images",
            'y_label': "Number of authors",
            'title': "Train Image Distribution by Authors"}}

    for col in ['Number_HSparrows', 'Author']:
        data = train_df[col]

        if col == 'Author':
            data = data.value_counts()
            x_ticks = range(1, len(data), 2)
        else:
            x_ticks = range(1, data.max()+1)

        fpath = (save_path / f'train_{col}_distribution.jpg'.lower()
                 if save_path else None)
        distribution_plot = draw_and_save_seaborn_plot(
            sns.histplot, data=data, **distribution_params[col],
            save_file_path=fpath, edgecolor='#1B6FA6', yaxis_grid=True,
            linewidth=2, binwidth=0.5, x_ticks=list(x_ticks))
        eda_plots.append(distribution_plot)
        logging.info(f"{distribution_params[col]['title']} plot is saved.")

    # Draw size plots
    for x, y, title, fname in [('image_width', 'image_height',
                                "Train Image Sizes", 'img'),
                               ('avg_bbox_width', 'avg_bbox_height',
                                "Average Train Bbox (House Sparrow) Sizes", 'avg_bbox')]:
        fpath = (save_path / f'train_{fname}_sizes.jpg' if save_path else None)
        size_plot = draw_and_save_seaborn_plot(
            sns.scatterplot, x=x, y=y, data=train_df, figsize=(6, 6), x_label=x,
            y_label=y, title=title, xaxis_grid=True, yaxis_grid=True, save_file_path=fpath)
        eda_plots.append(size_plot)
        logging.info(f"{title} plot is saved.")

    return eda_plots


if __name__ == '__main__':
    project_path = Path.cwd()
    param_config = get_param_config_yaml(project_path)
    _ = main(project_path, param_config, save_eda_plots=True)
