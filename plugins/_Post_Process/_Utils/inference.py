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
import csv
import argparse
import subprocess

from nnabla import logger
from nnabla.utils.cli import cli


def func(args):
    tmp_dir = os.path.splitext(args.output)[0]
    input_csv_file_name = os.path.join(tmp_dir, 'input.csv')
    if os.path.exists(args.output):
        os.remove(args.output)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Create input CSV file
    input_csv = [[], []]
    if os.path.exists(args.input_data) or ',' not in args.input_data:
        # File or scaler input
        input_csv[0].append(args.input_variable)
        input_csv[1].append(args.input_data)
    else:
        # Vector input
        for i, data in enumerate(args.input_data.split(',')):
            input_csv[0].append('{}__{}'.format(args.input_variable, i))
            input_csv[1].append(data)

    with open(input_csv_file_name, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(input_csv)

    # Run inference
    p = subprocess.call(
        ['python',
         cli.__file__,
         'forward',
         '-c',
         args.model,
         '-d',
         input_csv_file_name,
         '-o',
         os.getcwd(),
         '-f',
         args.output])

    if os.path.exists(args.output):
        logger.log(99, 'Inference completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Inference\n' +
        '\n' +
        'Perform inference on single data using trained model.\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m',
        '--model',
        help='path to model nnp file (model) default=results.nnp',
        required=True,
        default='results.nnp')
    parser.add_argument(
        '-v',
        '--input-variable',
        help='input variable name (variable) default=x',
        required=True)
    parser.add_argument(
        '-i', '--input-data', help='path to input data (file)', required=True)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output csv file (csv) default=inference.csv',
        required=True,
        default='inference.csv')
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
