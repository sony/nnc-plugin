# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import csv
import random
from tqdm import tqdm
import numpy as np

from nnabla import logger
from nnabla.utils.image_utils import imsave, imread


def get_variable_indexes(target_variables, source_variable_names):
    result = []
    if source_variable_names:
        for variable in source_variable_names.split(','):
            if variable in target_variables:
                result.append(target_variables.index(variable))
            else:
                logger.critical(
                    f'Variable {variable} is not found in the input CSV file.')
                raise
    return result


def load_image(src_file_name):
    im = imread(src_file_name)
    if len(im.shape) < 2 or len(im.shape) > 3:
        logger.warning(
            "Illegal image file format %s.".format(src_file_name))
        raise
    elif len(im.shape) == 3:
        # RGB image
        if im.shape[2] != 3:
            logger.warning(
                "The image must be RGB or monochrome.")
            csv_data.remove(data)
            raise
    else:
        # Monochrome image
        im = im.reshape((im.shape[0], im.shape[1], 1))
    return im


def func(args):
    # Open input CSV file
    logger.log(99, 'Loading input CSV file ...')
    with open(args.input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        table = [row for row in reader]
    variables = [v.split(':')[0] for v in header]
    input_csv_path = os.path.dirname(args.input_csv)

    # Settings for each variable
    image_index_index = get_variable_indexes(variables, args.image_index)[0]
    variable_index = get_variable_indexes(variables, args.input_variable)[0]
    height_variable_index = get_variable_indexes(
        variables, args.height_variable)[0]
    width_variable_index = get_variable_indexes(
        variables, args.width_variable)[0]
    patch_size_variable_index = get_variable_indexes(
        variables, args.patch_size_variable)[0]
    overlap_size_variable_index = get_variable_indexes(
        variables, args.overlap_size_variable)[0]
    top_variable_index = get_variable_indexes(variables, args.top_variable)[0]
    left_variable_index = get_variable_indexes(
        variables, args.left_variable)[0]
    inherit_col_indexes = get_variable_indexes(variables, args.inherit_cols)

    # Restore
    logger.log(99, 'Processing images ...')
    result_header = []
    for col_index in inherit_col_indexes:
        result_header.append(header[col_index])
    result_header.append(header[variable_index])
    result_table = []
    table.sort(key=lambda x: x[1])
    current_path = os.getcwd()
    output_path = os.path.dirname(args.output)

    # Input CSV file line loop
    last_index = -1
    tmp_lines = []

    def restore_image(lines):
        height = int(lines[0][height_variable_index])
        width = int(lines[0][width_variable_index])
        patch_size = int(lines[0][patch_size_variable_index])
        overlap_size = int(lines[0][overlap_size_variable_index])
        out_im = np.zeros((height, width, 3), dtype=np.uint8)
        result_line = []
        for col_index in inherit_col_indexes:
            result_line.append(lines[0][col_index])

        os.chdir(input_csv_path if input_csv_path else current_path)
        for line in lines:
            im = load_image(line[variable_index])
            top = int(line[top_variable_index]) + overlap_size
            left = int(line[left_variable_index]) + overlap_size
            patch_width = patch_height = patch_size - overlap_size * 2
            if top + patch_height > height:
                patch_height = height - top
            if left + patch_width > width:
                patch_width = width - left
            out_im[top:top + patch_height, left:left + patch_width, ::] = im[overlap_size:overlap_size +
                                                                             patch_height, overlap_size:overlap_size + patch_width, ::]
        out_im_csv_file_name = os.path.join(
            'restore_split_image',
            f'{last_index//1000:04}', f'{last_index % 1000:03}.png')
        out_im_file_name = os.path.join(output_path, out_im_csv_file_name)
        os.chdir(current_path)
        if not os.path.exists(os.path.dirname(out_im_file_name)):
            os.makedirs(os.path.dirname(out_im_file_name))
        imsave(out_im_file_name, out_im)
        result_line.append(out_im_csv_file_name)
        result_table.append(result_line)

    for line in tqdm(table):
        index = int(line[image_index_index])
        if index != last_index:
            if len(tmp_lines) > 0:
                restore_image(tmp_lines)
            tmp_lines = []
            last_index = index
        tmp_lines.append(line)

    if len(tmp_lines) > 0:
        restore_image(tmp_lines)

    logger.log(99, 'Saving CSV file...')
    with open(os.path.join(args.output), 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(result_header)
        writer.writerows(result_table)

    logger.log(99, 'Restore split image completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Restore Split Image\n\n' +
        'Restore large images from multiple images split into patches.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input-csv',
        help='dataset CSV file containing split images (csv) default=output_result.csv',
        required=True)
    parser.add_argument(
        '-n',
        '--image_index',
        help="image index variable in the dataset CSV (variable) default=index",
        required=True)
    parser.add_argument(
        '-v',
        '--input-variable',
        help="variables of the image to be processed in the dataset CSV (variable) default=y'",
        required=True)
    parser.add_argument(
        '-H',
        '--height-variable',
        help='image height variable (variable) default=y_original_height',
        required=True)
    parser.add_argument(
        '-W',
        '--width-variable',
        help='image width variable (variable) default=y_original_width',
        required=True)
    parser.add_argument(
        '-p',
        '--patch-size-variable',
        help='patch size variable (variable) default=y_patch_size',
        required=True)
    parser.add_argument(
        '-l',
        '--overlap-size-variable',
        help='overlap size variable (variable) default=y_overlap_size',
        required=True)
    parser.add_argument(
        '-y',
        '--top-variable',
        help='top coordinate variable (variable) default=y_top',
        required=True)
    parser.add_argument(
        '-x',
        '--left-variable',
        help='left coordinate variable (variable) default=y_left',
        required=True)
    parser.add_argument(
        '-o', '--output', help='output csv file (file) default=restored_images.csv', required=True)
    parser.add_argument(
        '-t', '--inherit-cols', help='variables to inherit from input CSV to output CSV (variables) default=# x')
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
