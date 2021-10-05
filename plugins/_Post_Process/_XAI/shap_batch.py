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
import numpy as np
import csv

import tqdm

import os
from nnabla import logger
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from utils.file import save_info_to_csv
from shap_utils.utils import get_executor, red_blue_map, gradient, get_interim_input, plot_shap


def func(args):

    executor = get_executor(args.model)

    # Prepare variable
    output_variable = list(executor.output_assign.keys())[0]

    # Data source
    data_iterator = (lambda: data_iterator_csv_dataset(
        uri=args.input,
        batch_size=1,
        shuffle=False,
        normalize=not executor.no_image_normalization,
        with_memory_cache=False,
        with_file_cache=False))

    # Prepare output
    data_output_dir = os.path.splitext(args.output)[0]

    # check
    csv_file = np.loadtxt(args.input, delimiter=",", dtype=str)
    header = [item.split(":")[0] for item in csv_file[0]]
    classes = csv_file[1:, header.index(args.label_variable)]
    num_classes = np.unique(classes).size

    output_size = output_variable.variable_instance.d.shape[1]
    is_binary_classification = num_classes != output_size

    # Data loop
    with data_iterator() as di:
        index = 0
        file_names = []
        while index < di.size:
            file_name = os.path.join(data_output_dir, '{:04d}'.format(
                                     index // 1000), '{}.png'.format(index))
            directory = os.path.dirname(file_name)
            try:
                os.makedirs(directory)
            except OSError:
                pass  # python2 does not support exists_ok arg
            # Load data
            data = di.next()
            im = data[di.variables.index(args.input_variable)]
            im = im.reshape((im.shape[1], im.shape[2], im.shape[3]))
            if is_binary_classification:
                label = 0
            else:
                label = data[di.variables.index(args.label_variable)]
                label = label.reshape((label.size,))
                if label.size > 1:
                    label = np.argmax(label)
                else:
                    label = label[0]
            if index == 0:
                pbar = tqdm.tqdm(total=di.size)

            plot_shap(model=args.model, X=im, label=label,
                      output=file_name, interim_layer=args.interim_layer,
                      num_samples=args.num_samples, data_iterator=data_iterator,
                      batch_size=args.batch_size, red_blue_map=red_blue_map, gradient=gradient,
                      get_interim_input=get_interim_input)

            file_names.append(file_name)
            index += 1
            pbar.update(1)

    pbar.close()
    save_info_to_csv(args.input, args.output, file_names, column_name='shap')

    logger.log(99, 'SHAP completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='SHAP(batch)\n'
                    '\n'
                    'A Unified Approach to Interpreting Model Predictions'
                    'Scott Lundberg, Su-In Lee'
                    'https://arxiv.org/abs/1705.07874\n'
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-in', '--input', help='path to input csv file (csv) default=output_result.csv', required=True, default='output_result.csv')
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-v1', '--input_variable', help='Variable to be processed (variable), default=x', required=True, default='x')
    parser.add_argument(
        '-v2', '--label_variable', help='Variable representing class index to visualize (variable) default=y', required=True, default='y')
    parser.add_argument(
        '-n', '--num_samples', help='number of samples N (int), default=100', required=True, type=int, default=100)
    parser.add_argument(
        '-b', '--batch_size', help=' batch size, default=50', required=True, type=int, default=50)
    parser.add_argument(
        '-il', '--interim_layer', help=' layer input to explain, default=0', required=True, type=int, default=0)
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv) default=shap.csv', required=True, default='shap.csv')
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
