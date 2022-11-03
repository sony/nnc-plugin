# Copyright 2022 Sony Group Corporation.
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
import argparse
import csv
import sys
import numpy as np
from nnabla import logger
from reject_option_based_classification import predict


def save_info_to_csv(input_path, output_path, file_names, column_name='gradcam', insert_pos=0):
    """
    save information to CSV file

    """
    with open(input_path, newline='') as f:
        rows = [row for row in csv.reader(f)]
    row0 = rows.pop(0)
    row0.insert(insert_pos, column_name)
    for i, file_name in enumerate(file_names):
        rows[i].insert(insert_pos, file_name)
    with open(output_path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(row0)
        writer.writerows(rows)


def extract_col(header, table, col):
    """
    extract columns
    """
    col_index = header.index(col)
    return [float(table[i][col_index]) for i in range(len(table))]


def enum_cols(header, name, col):
    if col not in header:
        logger.critical(
            'The column named {} specified by {} is not found in the input dataset'.format(
                col, name))
    else:
        logger.log(99, '{} : {}'.format(col, name))
    return col


def func(args):
    logger.log(99, 'Loading Evaluated Results ...')
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        in_header = next(reader)
        in_table = [[col.strip() for col in row]
                    for row in reader if len(row) > 0]

    with open(args.roc_params, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        roc_header = next(reader)
        roc_table = [[col.strip() for col in row]
                     for row in reader if len(row) > 0]

    output_col = enum_cols(in_header, 'output-variable', args.output_variable)
    privileged_col = enum_cols(in_header, 'privileged-variable', args.privileged_variable)
    unprivileged_col = enum_cols(in_header, 'unprivileged-variable', args.unprivileged_variable)
    roc_margin_col = enum_cols(roc_header, 'ROC', 'ROC Margin')
    roc_clf_threshold_col = enum_cols(roc_header, 'ROC', 'Classification Threshold')

    clf_out = np.asarray(extract_col(in_header, in_table, output_col)).flatten()
    privileged_variable = np.array(extract_col(in_header, in_table, privileged_col))
    unprivileged_variable = np.array(extract_col(in_header, in_table, unprivileged_col))
    roc_margin = float(extract_col(roc_header, roc_table, roc_margin_col)[0])
    roc_clf_threshold = float(extract_col(roc_header, roc_table, roc_clf_threshold_col)[0])
    # check privileged/unprivileged variables
    if not ((privileged_variable == 1.0) == (unprivileged_variable == 0.0)).all():
        logger.log(
            99, "Both privileged and unprivileged variable values should not be same")
        sys.exit(0)
    cond_privileged_group = privileged_variable == 1
    cond_unprivileged_group = unprivileged_variable == 1

    y_pred = predict(clf_out, roc_clf_threshold,
                     roc_margin, cond_privileged_group, cond_unprivileged_group)

    save_info_to_csv(args.input, args.output,
                     y_pred, column_name='ROC Predicted')



def main():
    parser = argparse.ArgumentParser(
        description='Reject Option-Based Classification Predict\n'
                    '\n'
                    'This plugin will obtain the fair predictions of the model using the ROC method, based on Reject '
                    'Option-Based Classification plugin results.\n\n' +
                    'Citation: \nKamiran, Faisal, Asim Karim, and Xiangliang Zhang. '
                    'Decision theory for discrimination-aware classification. In 2012 IEEE 12th '
                    'International Conference on Data Mining, pp. 924-929. IEEE, 2012. '
                    '\n' +
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input',
        help='path to input csv file, output result shown in the evaluation tab (csv) default=output_result.csv',
        required=True, default='output_result.csv')

    parser.add_argument(
        '-p', '--output_variable',
        help="predicted output variable, expect sigmoid output (variable) default=y'",
        required=True, default="y'")

    parser.add_argument(
        '-zp', '--privileged_variable',
        help='privileged variable from the input CSV file (variable)',
        required=True)
    parser.add_argument(
        '-zu', '--unprivileged_variable',
        help='unprivileged variable from the input CSV file (variable)',
        required=True)
    parser.add_argument(
        '-roc_params', '--roc_params',
        help='specify the Reject Option-Based Classification plugin processed output CSV file (csv) default=roc.csv',
        required=True, default='roc.csv')

    parser.add_argument(
        '-o', '--output', help='path to output file (csv) default=roc_predict.csv',
        required=True, default='roc_predict.csv')
    parser.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)
    logger.log(99, 'completed successfully.')


if __name__ == '__main__':
    main()
