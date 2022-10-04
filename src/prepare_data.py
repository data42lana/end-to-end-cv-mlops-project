from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold 

INFO_CSV_DATA_FILE = 'image_info.csv'
BBOX_CSV_DATA_FILE = 'bboxes/bounding_boxes.csv'

def load_data(csv_data_path):
    """Loads a csv file and returns it to pd.DataFrame."""
    return pd.read_csv(csv_data_path)

def stratified_group_train_test_split(data, stratification_basis, groups):
    """Stratified splits data into training and test sets,
    taking into account groups, and returns the corresponding indices."""
    split = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=0)
    train_ids, test_ids = next(split.split(X=data, y=stratification_basis, groups=groups))
    return train_ids, test_ids

def select_images_by_given_indices(image_series, indices):
    """Returns a pd.Series with selected images names by given indices."""
    return image_series.iloc[indices]

def expand_img_df_with_average_values_from_another_img_df(df1, df2,
                                                          selected_images, 
                                                          df2_columns_to_calculate_averages,
                                                          df1_image_name_column,
                                                          df2_image_name_column,
                                                          df2_columns_to_rename_in_new_df):
    """Expandes a DataFrame with selected images with columns from another DataFrame 
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
                      df2_columns_to_calculate_averages]
                 .groupby(df2_image_name_column)
                 .agg('mean')
                 .round()
                 .reset_index()
                 .rename(columns=rename_columns))

    new_expanded_df = (df1.loc[df1[df1_image_name_column].isin(selected_images)]
                          .merge(avg_df, on=df1_image_name_column, how='left'))
    return new_expanded_df

def main(project_path):
    data_path = project_path / 'data'
    raw_data_path = data_path / 'raw'
    img_info_df, img_bbox_df = [load_data(csv_data_file) for csv_data_file in [INFO_CSV_DATA_FILE, 
                                                                               BBOX_CSV_DATA_FILE]]
    train_ids, test_ids = stratified_group_train_test_split(img_info_df['Name'], 
                                                            img_info_df['Number_HSparrows'], 
                                                            img_info_df['Author'])

    for ids, fname in zip((train_ids, test_ids), ('train_555.csv', 'test_555.csv')): # !!!!!!!!
        sel_imgs = select_images_by_given_indices(img_info_df.Name, ids)
        cols_to_calculate_avg = ['bbox_width', 'bbox_height', 'image_width', 'image_height']
        expanded_df = expand_img_df_with_average_values_from_another_img_df(img_info_df, img_bbox_df,
                                                                            sel_imgs, cols_to_calculate_avg,
                                                                            'Name', 'image_name',
                                                                            cols_to_calculate_avg[:2])
        save_data_to_scv(expanded_df, data_path('prepared_222').mkdir(exist_ok=True), fname) # !!!!!!!

if __name__ == "__main__":
    project_path = Path(__file__).parent
    main(project_path)