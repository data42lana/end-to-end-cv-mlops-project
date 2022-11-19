"""This module checks csv data files and matches them with images."""

import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

CONFIG_PATH = 'configs/config.yaml'

def get_data_type_arg_parser():
    """Returns a argument parser object with a type of data."""
    parser = argparse.ArgumentParser(
        description='Specify a type of data to check.',
        add_help=False)
    parser.add_argument('--check_data_type', type=str, choices=['raw', 'new'],
                        default='raw', help='check raw or new data')
    return parser

def check_that_two_sorted_lists_are_equal(l1, l2, passed_message=''):
    """Returns a dictionary of the validation status with a list 
    of non-matching elements or the number of duplicates, if any."""
    l1 = sorted(l1)
    l2 = sorted(l2)
    
    if l1 == l2:
        return {'PASSED': passed_message}
    elif (len(set(l1)) != len(l1)) or (len(set(l2)) != len(l2)):
        return {'WARNING: Duplicates!': len(l1 + l2) - len(set(l1)) - len(set(l2))}
    else:        
        not_match = list(set(l1) ^ set(l2))
        return {'FAILED': not_match}

def check_that_series_is_less_than_or_equal_to(s1, other, comparison_sign, passed_message=''):
    """Returns a dictionary of the validation status with indices with incorrect values, if any.
    
    Parameters:
        s1 (pd.Series): a object to be compared
        other (pd.Series or scalar value): a object to compare
        comparison_sign (str): must be one of '==', '<='. Otherwise raises ValueError
        passed_message (str): a message that describes a passage of the check.
    """  
    comp_series_result = 0

    if comparison_sign == '==':
        comp_series_result  = s1.eq(other)
    elif comparison_sign == '<=':
        comp_series_result  = s1.le(other) 
    else:
        raise ValueError()       

    if comp_series_result.sum() == s1.shape[0]:
        return {'PASSED': passed_message}
    else:
        return {'FAILED': s1[~comp_series_result].index}
        
def main(project_path, check_data_type):
    """Checks csv data files and matches them with images."""
    project_path = Path(project_path)

    # Get image data paths from a configuration file
    with open(project_path / CONFIG_PATH) as f:
        config = yaml.safe_load(f)        
    img_data_paths = config['image_data_paths'] if check_data_type == 'raw' else config['new_image_data_paths']
        
    # Get a list of image names
    img_names = [img.parts[-1] for img in (project_path / img_data_paths['images']).iterdir()]

    img_info_df, img_bbox_df = [
        pd.read_csv(project_path / img_data_paths[csv_file]) for csv_file in ['info_csv_file', 
                                                                              'bboxes_csv_file']]
    # Create a dict of validation results for summary report
    validation_results = {}

    # Check whether names of available images are identical to names in csv files
    for img_name_list, csv_data_file in zip([img_info_df.Name.to_list(), img_bbox_df.image_name.unique()],
                                            ['info_csv_file', 'bbox_csv_file']):
        validation_results["Image Name Match Check: " + csv_data_file] = check_that_two_sorted_lists_are_equal(
            img_name_list, img_names,
            passed_message="The image names in the file correspond to the available images.")

    # Check the correctness of bounding box parameters
    for bb_param, img_param in [('bbox_x', 'width'), 
                                ('bbox_y', 'height')]:
        
        for add_bb_param in ('', 'bbox_' + img_param):
            add_values = 0
            img_name_param = 'image_' + img_param
            check_param_name = bb_param        

            if add_bb_param:
                add_values = img_bbox_df[add_bb_param]
                check_param_name = ' + '.join([check_param_name, add_bb_param])

            comp_bbox_img_param_result = check_that_series_is_less_than_or_equal_to(
                                             img_bbox_df[bb_param].add(add_values), 
                                             img_bbox_df[img_name_param], '<=', 
                                             passed_message=f"Correct: ({check_param_name}) <= {img_name_param}.") 
            check_name = f"Bbox Parameter Correctness Check: " + check_param_name    
            validation_results[check_name] = comp_bbox_img_param_result

    # Check the correctness of image parameters
    uniq_img_param_df = (img_bbox_df[['image_name', 'image_width', 'image_height']]
                             .groupby('image_name', group_keys=True)
                             .nunique())

    for img_param in ('image_width', 'image_height'):
        validation_results[f"Image Parameter Correctness Check: " + img_param] = check_that_series_is_less_than_or_equal_to(
            uniq_img_param_df[img_param], 1, '==',
            passed_message="One unique value for each image.")

    # Check if the number of house sparrows and the number of bounding boxes match
    number_hsparrows = (img_info_df[['Name', 'Number_HSparrows']].sort_values(by='Name')
                                                                 .set_index('Name')
                                                                 .squeeze())
    number_bboxes = img_bbox_df['image_name'].sort_values().value_counts(sort=False)

    validation_results["Number Match Check: Number_HSparrows vs image_name"] = check_that_series_is_less_than_or_equal_to(
        number_hsparrows, number_bboxes, '==', passed_message="The numbers match.")
    
    # Save validation results to a file
    fname = f'{check_data_type}_csv_file_check_results.txt'
    file_save_path = project_path / f'data_checks/data_check_results/{fname}'
    file_save_path.parent.mkdir(parents=True, exist_ok=True)
    file_save_path.write_text(str(validation_results))
    print("[INFO]: Check results are saved.")
    
if __name__ == '__main__':
    project_path = Path(__file__).parent.parent
    data_type_parser = argparse.ArgumentParser('Image data csv file check script.', 
                                               parents=[get_data_type_arg_parser()])
    img_data_type = data_type_parser.parse_args()
    main(project_path, img_data_type.check_data_type)