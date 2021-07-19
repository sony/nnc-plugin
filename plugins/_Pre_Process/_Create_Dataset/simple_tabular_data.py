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
import copy
import logging
from tqdm import tqdm

import numpy as np
from nnabla import logger


def func(args):
    # Prepare logger
    if args.log_file_output is not None:
        handler = logging.FileHandler(
            os.path.join(
                args.output_dir,
                args.log_file_output))
        logger.addHandler(handler)

    # Load input dataset
    logger.log(99, 'Loading original dataset ...')
    with open(args.input, 'r', encoding=args.encoding) as f:
        reader = csv.reader(f)
        in_header = next(reader)
        in_table = [[col.strip() for col in row]
                    for row in reader if len(row) > 0]

    # Load preprocessing parameters
    in_preprocessing_param = None
    if args.preprocess_param_input is not None:
        logger.log(99, 'Loading preprocessing parameters ...')
        with open(args.preprocess_param_input, 'r') as f:
            reader = csv.reader(f)
            in_preprocessing_param = [row for row in reader]
        in_preprocessing_param_header = [row[0]
                                         for row in in_preprocessing_param]

    # Decode cols
    def enum_cols(name, cols):
        if cols is None:
            return []
        result = cols.split(',')
        for col in result:
            if col not in in_header:
                logger.critical(
                    'The column named {} specified by {} is not found in the input dataset'.format(
                        col, name))
        return result

    comment_cols = enum_cols('comment-cols', args.comment_cols)
    include_cols = enum_cols('include-variables', args.include_variables)
    exclude_cols = enum_cols('exclude-variables', args.exclude_variables)
    objective_cols = enum_cols('objective-variables', args.objective_variables)

    if len(include_cols):
        explanatory_cols = include_cols
    else:
        explanatory_cols = copy.deepcopy(in_header)
        for ext_col in comment_cols + exclude_cols + objective_cols:
            if ext_col in explanatory_cols:
                explanatory_cols.remove(ext_col)

    logger.log(99, 'Explanatory Cols : {}'.format(explanatory_cols))
    logger.log(99, 'Objective Cols : {}'.format(objective_cols))

    # Prepare output table
    out_header = []
    out_table = [[] for _ in range(len(in_table))]
    out_preprocessing_param = []

    # Add comment cols to the output table
    for comment_col in comment_cols:
        out_header.append('# ' + comment_col)
        col_index = in_header.index(comment_col)
        for i in range(len(in_table)):
            out_table[i].append(in_table[i][col_index])

    # Add explanatory cols to the output table
    def extract_col(col):
        col_index = in_header.index(col)
        return [in_table[i][col_index] for i in range(len(in_table))]

    def check_numeric(values):
        result = []
        with_value = []
        for value in values:
            if value == '':
                result.append(0)
                with_value.append(False)
            try:
                result.append(float(value))
                with_value.append(True)
            except BaseException:
                return False, [], []
        return True, result, with_value

    col_index = 0
    for col in explanatory_cols + objective_cols:
        variable_name = 'y' if col in objective_cols else 'x'
        if len(objective_cols) and objective_cols[0] == col:
            col_index = 0

        values = extract_col(col)
        is_numeric, numeric_values, with_value = check_numeric(values)

        if in_preprocessing_param is None:
            if is_numeric:
                logger.log(
                    99, '{} is determined to be a numeric variable'.format(col))
                if args.standardize:
                    vec = np.array([value for value, wv in zip(
                        numeric_values, with_value) if wv])
                    mean = vec.mean()
                    std = vec.std()
                    logger.log(
                        99, '    Standardized with mean {} and std {}.'.format(
                            mean, std))
                    values = [
                        (value - mean) / std if wv else value for value,
                        wv in zip(
                            numeric_values,
                            with_value)]
                else:
                    mean = 0.0
                    std = 1.0
            else:
                # Category
                logger.log(
                    99, '{} is determined to be a category variable'.format(col))
                categories = sorted(set(values))
        else:
            # Use preprocessing param file for preprocessing
            if col not in in_preprocessing_param_header:
                logger.critical(
                    'The column named {} is not found in the preprocessing parameter file {}'.format(
                        col, args.preprocess_param_input))
            param_index = in_preprocessing_param_header.index(col)
            if in_preprocessing_param[param_index][1] == 'numeric':
                # Numeric
                is_numeric = True
                mean = float(in_preprocessing_param[param_index][2])
                std = float(in_preprocessing_param[param_index][3])
                if mean != 1.0 or std != 0.0 and len(
                        numeric_values) == len(values):
                    values = [
                        (value - mean) / std if wv else value for value,
                        wv in zip(
                            numeric_values,
                            with_value)]
            else:
                # Category
                is_numeric = False
                categories = in_preprocessing_param[param_index][2:]

        if is_numeric:
            out_preprocessing_param.append([col, 'numeric', mean, std])
            out_header.append(
                '{}__{}:{}'.format(
                    variable_name,
                    col_index,
                    col))
            col_index += 1
            for i in range(len(in_table)):
                try:
                    out_table[i].append(values[i])
                except BaseException:
                    logger.critical(
                        'i={}, col={}, values={}.'.format(
                            i, col, values))
        else:
            # Category
            param = [col, 'category']
            param.extend(categories)
            out_preprocessing_param.append(param)
            if col in objective_cols:
                out_header.append('y__{}:{}'.format(col_index, col))
                col_index += 1
                for i in range(len(in_table)):
                    try:
                        out_table[i].append(categories.index(values[i]))
                    except BaseException:
                        logger.critical(
                            'Unknown category {} found in {}, index {}.'.format(
                                values[i], col, i))
            else:
                out_header.extend(['x__{}:{}={}'.format(
                    j + col_index, col, category) for j, category in enumerate(categories)])
                col_index += len(categories)
                for i in range(len(in_table)):
                    out_table[i].extend(
                        [1 if categories[j] == values[i] else 0 for j in range(len(categories))])

    # Output preprocessing parameters
    if args.preprocess_param_output is not None:
        with open(os.path.join(args.output_dir, args.preprocess_param_output), 'w', newline="\n", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(out_preprocessing_param)

    # Output dataset
    logger.log(99, 'Creating NNC dataset ...')

    if args.shuffle:
        random.shuffle(out_table)

    logger.log(99, 'Saving output file 1 ...')
    with open(os.path.join(args.output_dir, args.output_file1), 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        writer.writerows(out_table[: int(len(out_table) * args.ratio1) // 100])

    if args.output_file2 is not None and args.ratio2 > 0:
        logger.log(99, 'Saving output file 2 ...')
        with open(os.path.join(args.output_dir, args.output_file2), 'w', newline="\n", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(out_header)
            writer.writerows(
                out_table[int(len(out_table) * args.ratio1) // 100:])

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Simple Tabular Dataset\n\n' +
        'Convert a tabular dataset to NNC format dataset.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input', help='tabular dataset (csv)', required=True)
    parser.add_argument(
        '-E',
        '--encoding',
        help='input text file encoding (text), default=utf-8-sig',
        default='utf-8-sig')
    parser.add_argument(
        '-c',
        '--comment-cols',
        help='specify columns to be included as a comment separated by commas (text)')
    parser.add_argument(
        '-n',
        '--include-variables',
        help='specify variables to be included in the explanatory variables separated by commas (text)')
    parser.add_argument(
        '-e',
        '--exclude-variables',
        help='specify variables to be excluded in the explanatory variables separated by commas (text)')
    parser.add_argument(
        '-b',
        '--objective-variables',
        help='specify variables to be included in the objective variables separated by commas (text)')
    parser.add_argument(
        '-t',
        '--standardize',
        help='standardize continuous values to mean 0 standard diviation 1 (bool) default=True',
        action='store_true')
    parser.add_argument(
        '-p',
        '--preprocess-param-input',
        help='preprocessing parameter file input (csv)')
    parser.add_argument(
        '-r',
        '--preprocess-param-output',
        help='preprocessing parameter file output (csv), default=preprocessing_parameters.csv')
    parser.add_argument(
        '-o', '--output-dir', help='output directory (dir)', required=True)
    parser.add_argument(
        '-g',
        '--log-file-output',
        help='log file output (file), default=log.txt')
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
        help='output file 1 ratio as a percentage (int) default=100',
        type=float,
        required=True)
    parser.add_argument(
        '-f2',
        '--output_file2',
        help='output file name 2 (csv) default=test.csv',
        default='test.csv')
    parser.add_argument(
        '-r2',
        '--ratio2',
        help='output file 2 ratio as a percentage (int) default=0',
        type=float,
        default=0)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
