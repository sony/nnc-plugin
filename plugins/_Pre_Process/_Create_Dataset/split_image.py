# Copyright 2021,2022,2023 Sony Group Corporation.
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
    logger.log(99, 'Loading original dataset ...')
    with open(args.input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        table = [row for row in reader]
    variables = [v.split(':')[0] for v in header]
    input_csv_path = os.path.dirname(args.input_csv)
    image_path = 'split_' + \
        os.path.splitext(os.path.basename(args.input_csv))[0]

    # Settings for each variable
    variable_indexes = [[], []]
    variable_indexes[0] = get_variable_indexes(variables, args.input_variable1)
    variable_indexes[1] = get_variable_indexes(variables, args.input_variable2)
    patch_sizes = [args.patch_size1, args.patch_size2]
    overlap_sizes = [args.overlap_size1, args.overlap_size2]

    # Add new cols to the header
    extra_header = []
    header.append('index')
    for i in range(2):
        for vi in variable_indexes[i]:
            extra_header.append(header[vi])
            header.extend([f'{variables[vi]}_original_height', f'{variables[vi]}_original_width', f'{variables[vi]}_patch_size',
                           f'{variables[vi]}_overlap_size', f'{variables[vi]}_top', f'{variables[vi]}_left'])
    header.extend(extra_header)

    # Comment out original cols
    for i, h in enumerate(header):
        if (i in variable_indexes[0] or i in variable_indexes[1]):
            header[i] = '# ' + header[i]

    # Add original data index to the table
    for i in range(len(table)):
        table[i].append(i)

    # Convert
    logger.log(99, 'Processing images ...')
    result_table = [[], []]
    os.chdir(input_csv_path)
    output_path = args.output_dir
    if args.shuffle:
        random.shuffle(table)

    # Input CSV file line loop
    for org_data_index, line in enumerate(tqdm(table)):
        result_lines = []
        extra_lines = []
        im_output_path = os.path.join(
            output_path, image_path, f'{org_data_index//1000:04}')
        if not os.path.exists(im_output_path):
            os.makedirs(im_output_path)
        for i in range(2):  # Variable 1 and 2
            for vi in variable_indexes[i]:
                base_size = patch_sizes[i] - overlap_sizes[i] * 2
                im = load_image(line[vi])
                line[vi] = os.path.join(input_csv_path, line[vi])
                patch_num_x = (im.shape[1] - 1) // base_size + 1
                patch_num_y = (im.shape[0] - 1) // base_size + 1
                patch_index = 0
                # Patch loop
                for py in range(patch_num_y):
                    top = org_top = -overlap_sizes[i] + py * base_size
                    if top >= 0:
                        y_size = patch_sizes[i]
                        y_offset = 0
                    else:
                        y_offset = -top
                        y_size = patch_sizes[i] - y_offset
                        top = 0
                    if top + y_size > im.shape[0]:
                        y_size = im.shape[0] - top
                    for px in range(patch_num_x):
                        left = org_left = - overlap_sizes[i] + px * base_size
                        if left >= 0:
                            x_size = patch_sizes[i]
                            x_offset = 0
                        else:
                            x_offset = -left
                            x_size = patch_sizes[i] - x_offset
                            left = 0
                        if left + x_size > im.shape[1]:
                            x_size = im.shape[1] - left

                        # Crop image
                        out_im = np.zeros(
                            (patch_sizes[i], patch_sizes[i], 3), dtype=np.uint8)
                        crop_im = im[top:top + y_size, left:left + x_size, ::]
                        out_im[y_offset:y_offset + crop_im.shape[0],
                               x_offset:x_offset + crop_im.shape[1], ::] = crop_im

                        # Add line
                        if len(result_lines) <= patch_index:
                            result_lines.append([])
                            extra_lines.append([])
                        out_im_csv_file_name = os.path.join(
                            image_path,
                            f'{org_data_index//1000:04}', f'{variables[vi]}_{org_data_index % 1000:03}_{patch_index:04}.png')
                        out_im_file_name = os.path.join(
                            output_path, out_im_csv_file_name)
                        imsave(out_im_file_name, out_im)
                        result_lines[patch_index].extend(
                            [out_im_csv_file_name])
                        extra_lines[patch_index].extend(
                            [im.shape[0], im.shape[1], patch_sizes[i], overlap_sizes[i], org_top, org_left])
                        patch_index += 1
        result_table[0 if org_data_index < (len(table) * args.ratio1) // 100 else 1].extend(
            [line + e + r for r, e in zip(result_lines, extra_lines)])

    logger.log(99, 'Saving output file 1 ...')
    with open(os.path.join(args.output_dir, args.output_file1), 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        if args.shuffle:
            random.shuffle(result_table[0])
        writer.writerows(result_table[0])

    if args.output_file2 is not None and len(result_table[1]):
        logger.log(99, 'Saving output file 2 ...')
        with open(os.path.join(args.output_dir, args.output_file2), 'w', newline="\n", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(result_table[1])

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Split Image\n\n' +
        'Split a large image into multiple patch images of a size suitable for signal processing by deep learning.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input-csv',
        help='dataset CSV file containing input and output images (csv)',
        required=True)
    parser.add_argument(
        '-v1',
        '--input-variable1',
        help='variables of the image to be processed in the dataset CSV (variable) default=x',
        required=True)
    parser.add_argument(
        '-p1',
        '--patch-size1',
        help='specify patch size in pixels (int) default=64',
        required=True,
        type=int)
    parser.add_argument(
        '-l1',
        '--overlap-size1',
        help='specify the size of the patch to be duplicated in pixels (int) default=8',
        required=True,
        type=int)
    parser.add_argument(
        '-v2',
        '--input-variable2',
        help='variables of the image to be processed in the dataset CSV (variable) default=y')
    parser.add_argument(
        '-p2',
        '--patch-size2',
        help='specify patch size in pixels (int) default=64',
        type=int)
    parser.add_argument(
        '-l2',
        '--overlap-size2',
        help='specify the size of the patch to be duplicated in pixels (int) default=8',
        type=int)
    parser.add_argument(
        '-o', '--output-dir', help='output dir (dir)', required=True)
    parser.add_argument(
        '-s',
        '--shuffle',
        help='shuffle (bool) default=True',
        action='store_true')
    parser.add_argument(
        '-f1',
        '--output_file1',
        help='output file name 1 (csv) default=train.csv',
        required=True,
        default='train.csv')
    parser.add_argument(
        '-r1',
        '--ratio1',
        help='output file ratio 1 (int) default=100',
        type=int,
        required=True)
    parser.add_argument(
        '-f2',
        '--output_file2',
        help='output file name 2 (csv) default=test.csv',
        default='test.csv')
    parser.add_argument(
        '-r2',
        '--ratio2',
        help='output file ratio 2 (int) default=0',
        type=int,
        default=0)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
