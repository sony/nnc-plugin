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
import os
import shutil
import csv
import argparse
import subprocess
import zipfile

import google.protobuf.text_format as text_format

from nnabla import logger
from nnabla.utils.cli import cli
from nnabla.utils import nnabla_pb2


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

    # Copy model to the tmp dir
    model_file = os.path.join(tmp_dir, args.model)
    shutil.copyfile(args.model, model_file)

    # Edit model (replace the output variable of executor with the args.layer_name)
    proto = nnabla_pb2.NNablaProtoBuf()
    with zipfile.ZipFile(model_file) as z:
        for name in z.namelist():
            ext = os.path.splitext(name)[1].lower()
            if ext == ".nntxt" or ext == ".prototxt":
                with z.open(name, mode='r') as f:
                    text_format.Merge(f.read(), proto)
                for e in proto.executor:
                    while len(e.output_variable) > 1:
                        del e.output_variable[1]
                    e.output_variable[0].variable_name = args.layer_name
                    e.output_variable[0].data_name = args.layer_name
                edited_nntxt = os.path.join(tmp_dir, name)
                break
    with open(edited_nntxt, mode='w') as f:
        text_format.PrintMessage(proto, f)
    with zipfile.ZipFile(model_file, 'a') as z:
        z.write(edited_nntxt)

    # Run Forward
    p = subprocess.call(
        ['python',
         cli.__file__,
         'forward',
         '-c',
         model_file,
         '-d',
         input_csv_file_name,
         '-o',
         tmp_dir,
         '-f',
         os.path.join('..', args.output)])

    # Replace data path
    replace_path = '.' + os.path.sep
    with open(args.output, encoding='utf_8_sig') as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            for i in range(len(row)):
                if row[i][:len(replace_path)] == replace_path:
                    row[i] = os.path.join(tmp_dir, row[i][len(replace_path):])
            rows.append(row)
    with open(args.output, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(rows)

    # delete temporary model
    os.remove(model_file)

    if os.path.exists(args.output):
        logger.log(99, 'Feature Visualization completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Feature Visualization\n' +
        '\n' +
        'Visualize intermediate activation of the model with a single piece of data.\n' +
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
        '-l', '--layer-name', help='layer name to visualize (text) default=Convolution', required=True)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output csv file (csv) default=feature_visualization.csv',
        required=True,
        default='feature_visualization.csv')
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
