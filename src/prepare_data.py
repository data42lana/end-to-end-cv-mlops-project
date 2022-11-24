"""This module creates training and test datasets from raw data."""

from pathlib import Path
import logging

import yaml
import numpy as np
import pandas as pd

from utils import stratified_group_train_test_split

# Set reproducibility
SEED = 0

CONFIG_PATH = 'configs/config.yaml'


def expand_img_df_with_average_values_from_another_img_df(df1, df2,
                                                          selected_images, 
                                                          df2_columns_to_calculate_averages,
                                                          df1_image_name_column,
                                                          df2_image_name_column,
                                                          df2_columns_to_rename_in_new_df):
    """Expands a DataFrame with selected images with columns from another DataFrame 
    with averages calculated for each group of these images.

    Parameters:
        df1 (pd.DataFrame): a pd.DataFrame object to expand
        df2 (pd.DataFrame): a pd.DataFrame object to calculate averages. 
        selected_images (list): images for which average values from df2 will be calculated
        df2_columns_to_calculate_averages (list): df2 columns to calculate average values
        df1_image_name_column (str): a df1 column with image names to merge
        df2_image_name_column (str): a df2 column with image names to merge
        df2_columns_to_rename_in_new_df (list): df2 columns with averages to be renamed
    
    Returns:
        a new expanded pd.DataFrame object.
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

def main(project_path):
    """Creates training and test csv data files."""
    project_path = Path(project_path)
    logging.basicConfig(level=logging.INFO)

    # Get image data paths from a configuration file
    with open(project_path / CONFIG_PATH) as f:
        config = yaml.safe_load(f)  
    img_data_paths = config['image_data_paths']
    
    # Split data into training and test sets
    img_info_df, img_bbox_df = [
        pd.read_csv(project_path / img_data_paths[csv_data_file]) for csv_data_file in ['info_csv_file', 
                                                                                        'bboxes_csv_file']]
    train_ids, test_ids = stratified_group_train_test_split(img_info_df['Name'], 
                                                            img_info_df['Number_HSparrows'], 
                                                            img_info_df['Author'], 
                                                            SEED)
    # Create training and test csv files
    for ids, fpath in zip((train_ids, test_ids), ('train_csv_file', 'test_csv_file')):
        fpath = project_path / img_data_paths[fpath]
        fpath.parent.mkdir(exist_ok=True)

        sel_imgs = img_info_df.Name.iloc[ids]
        cols_to_calculate_avg = ['bbox_width', 'bbox_height', 'image_width', 'image_height']
        expanded_df = expand_img_df_with_average_values_from_another_img_df(img_info_df, img_bbox_df,
                                                                            sel_imgs, cols_to_calculate_avg,
                                                                            'Name', 'image_name',
                                                                            cols_to_calculate_avg[:2])
        expanded_df.to_csv(fpath, index=False)
        logging.info("Train and test csv files are saved.")

if __name__ == '__main__':
    project_path = Path(__file__).parent.parent
    main(project_path)