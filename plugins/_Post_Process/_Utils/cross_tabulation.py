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
import argparse
import csv
import numpy as np
from nnabla import logger


def extract_col(header, table, col_name):
    col_index = [j for j, col in enumerate(
        header)if col_name == col.split('__')[0].split(':')[0]]

    if len(col_index) == 0:
        logger.critical(
            'Variable {} is not found in the dataset.'.format(col_name))
        return []
    elif len(col_index) == 1:
        result = [line[col_index[0]] for line in table]
        if any('.' in cell for cell in result):
            # float value -> binary score
            logger.log(
                99, 'The probability of {} has been binarized.'.format(col_name))
            return [1 if float(cell) >= 0.5 else 0 for cell in result]
        else:
            return [int(cell) for cell in result]
    else:
        logger.log(
            99,
            '{} has been converted to the index of the largest value of {}'.format(
                col_name,
                col_name))
        return [int(np.argmax([float(line[j]) for j in col_index]))
                for line in table]


def func(args):
    # print(args.output_in_ratio)
    col_names = [args.variable1, args.variable2]
    if args.variable2_eval is not None:
        col_names.append(args.variable2_eval)

    with open(args.input, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        table = [row for row in reader]

    col_names = list(set(col_names))
    cols = {}
    for col_name in col_names:
        col = extract_col(header, table, col_name)
        if len(col):
            cols[col_name] = col
        else:
            # Column not found
            return

    num_row = np.max(cols[args.variable1]) + 1
    num_column = np.max(cols[args.variable2]) + 1
    if args.variable2_eval is not None:
        # Evaluate whether it is correct or incorrect
        new_col_name = args.variable2 + ' = ' + args.variable2_eval
        cols[new_col_name] = [int(v1 != v2) for v1, v2 in zip(
            cols[args.variable2], cols[args.variable2_eval])]
        output_col_names = ['Correct', 'Incorrect']
        args.variable2 = new_col_name
        num_column = 2
    else:
        output_col_names = [
            args.variable2 +
            ' = ' +
            str(j) for j in range(num_column)]

    with open(args.output, 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Row variable', args.variable1])
        writer.writerow(['Column variable', args.variable2])
        writer.writerow([])
        writer.writerow([''] + output_col_names)

        for i in range(num_row):
            line = []
            sum = 0
            for j in range(num_column):
                num = [v1 == i and v2 == j for v1, v2 in zip(
                    cols[args.variable1], cols[args.variable2])].count(True)
                line.append(int(num))
                sum += num
            if args.output_in_ratio and sum > 0:
                line = [cell * 1.0 / sum for cell in line]
            writer.writerow([args.variable1 + ' = ' + str(i)] + line)
    logger.log(99, 'Cross tabulation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Cross Tabulation\n\n' +
        'Count the number of data according to the value of the variable\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='path to input csv file (csv) default=output_result.csv',
        required=True,
        default='output_result.csv')
    parser.add_argument(
        '-v1',
        '--variable1',
        help='variable name for row (variable) default=y',
        required=True,
        default='y')
    parser.add_argument(
        '-v2',
        '--variable2',
        help='variable name for col (variable) default=y',
        required=True,
        default='y')
    parser.add_argument(
        '-v2e',
        '--variable2_eval',
        help="variable name indicating the evaluation result (variable) default=y'")
    parser.add_argument(
        '-r',
        '--output_in_ratio',
        help='Output in ratio (bool), default=True',
        action='store_true')
    parser.add_argument(
        '-o',
        '--output',
        help='path to output csv file (csv) default=cross_tabulation.csv',
        required=True,
        default='cross_tabulation.csv')
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
