"""This module creates training and test datasets from raw data."""

import logging
from pathlib import Path

import pandas as pd

from src.utils import get_param_config_yaml, stratified_group_train_test_split

logging.basicConfig(level=logging.INFO, filename='app.log',
                    format="[%(levelname)s]: %(message)s")

# Set reproducibility
SEED = 0


def expand_img_df_with_average_values_from_another_img_df(df1, df2,
                                                          selected_images,
                                                          df2_columns_to_calculate_averages,
                                                          df1_image_name_column,
                                                          df2_image_name_column,
                                                          df2_columns_to_rename_in_new_df):
    """Expand a DataFrame with selected images with columns from another DataFrame
    with averages calculated for each group of these images.

    Parameters
    ----------
    df1: pd.DataFrame
        A pd.DataFrame object to expand.
    df2: pd.DataFrame
        A pd.DataFrame object to calculate averages.
    selected_images: list
        Images for which average values from df2 will be calculated.
    df2_columns_to_calculate_averages: list
        df2 columns to calculate average values.
    df1_image_name_column: str
        A df1 column with image names to merge.
    df2_image_name_column: str
        A df2 column with image names to merge.
    df2_columns_to_rename_in_new_df: list
        df2 columns with averages to be renamed.

    Return
    -------
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


def main(project_path, param_config):
    """Create training and test CSV data files."""
    # Get image data paths from configurations
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

    # Create training and test CSV files
    for ids, fpath in zip((train_ids, test_ids), ('train_csv_file', 'test_csv_file')):
        fpath = project_path / img_data_paths[fpath]
        fpath.parent.mkdir(exist_ok=True)

        sel_imgs = img_info_df.Name.iloc[ids]
        cols_to_calculate_avg = ['bbox_width', 'bbox_height', 'image_width', 'image_height']
        expanded_df = expand_img_df_with_average_values_from_another_img_df(
            img_info_df, img_bbox_df, sel_imgs, cols_to_calculate_avg,
            'Name', 'image_name', cols_to_calculate_avg[:2])
        expanded_df.to_csv(fpath, index=False)
        logging.info("Train and test csv files are saved.")


if __name__ == '__main__':
    project_path = Path.cwd()
    param_config = get_param_config_yaml(project_path)
    main(project_path, param_config)
