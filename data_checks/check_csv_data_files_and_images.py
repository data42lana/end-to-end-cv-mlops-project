"""This module checks csv data files and matches them with images."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_PATH = 'data/raw'
IMG_DIR = 'images'
SAVE_CHECK_RESULT_DIR = 'data_checks/data_check_results'

INFO_CSV_DATA_FILE = 'image_info.csv'
BBOX_CSV_DATA_FILE = 'bboxes/bounding_boxes.csv'

def get_path_args_parser():
    """Returns a argument parser object with relative data paths."""
    parser = argparse.ArgumentParser(
        description='Specify a relative path to files and a directory with corresponding images to check.',
        add_help=False)
    parser.add_argument('--info_csv_file_path', type=str, 
                        default='/'.join([RAW_DATA_PATH, INFO_CSV_DATA_FILE]),
                        help='a relative path from a project directory to a info csv file')
    parser.add_argument('--bbox_csv_file_path', type=str,
                        default='/'.join([RAW_DATA_PATH, BBOX_CSV_DATA_FILE]),
                        help='a relative path from a project directory to a bbox csv file')
    parser.add_argument('--image_dir_path', type=str, 
                        default='/'.join([RAW_DATA_PATH, IMG_DIR]),
                        help='a relative path from a project directory to a directory with corresponding images')
    parser.add_argument('--save_validation_result_path', type=str, 
                        default='/'.join([SAVE_CHECK_RESULT_DIR, 'csv_file_check_results.txt']),
                        help='a relative path from a project directory to a file to save check results')
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
        comparison_sign (str): must be one of '==', '<='. Otherwise raises ValueError.
        passed_message (str): a message that describes a passage of the check
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
        
def main(project_path, img_data_path_args):
    """Checks csv data files and matches them with images."""
    project_path = Path(project_path)
    # Get a list of image names
    img_names = [img.parts[-1] for img in (project_path / img_data_path_args.image_dir_path).iterdir()]

    img_info_df, img_bbox_df = [pd.read_csv(project_path / csv_file) for csv_file in [img_data_path_args.info_csv_file_path, 
                                                                                      img_data_path_args.bbox_csv_file_path]]
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
    file_save_path = project_path / img_data_path_args.save_validation_result_path
    file_save_path.touch(exist_ok=True)
    file_save_path.write_text(str(validation_results))

    
if __name__ == '__main__':
    project_path = Path(__file__).parent.parent
    data_path_parser = argparse.ArgumentParser('Image data csv file check script.', parents=[get_path_args_parser()])
    img_data_path_args = data_path_parser.parse_args()
    main(project_path, img_data_path_args)