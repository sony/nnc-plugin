# Copyright 2021,2022 Sony Group Corporation.
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

import numpy as np
import argparse
import csv
from nnabla import logger
from shap_tabular_utils.calculate import KernelSHAP, check_datasize
from shap_tabular_utils.visualize import visualize


def func(args):
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        _ = next(reader)
        X = np.array([[float(r) for r in row] for row in reader])[:, :-1]
    with open(args.train, 'r') as f:
        reader = csv.reader(f)
        feature_names = next(reader)[:-1]
        data = np.array([[float(r) for r in row] for row in reader])[:, :-1]
    is_executable_size = check_datasize(data)
    if not is_executable_size:
        raise Exception(
            f'dataset size of {data.shape} might be too large to execute shap computation.')

    kernel_shap = KernelSHAP(data, args.model, X, args.alpha)
    shap_values, expected_values = kernel_shap.calculate_shap()
    shap_value_class = shap_values[args.class_index]
    with open(args.output, 'w', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(['Index'] + feature_names)
        for i, result in enumerate(shap_value_class):
            writer.writerow(
                [str(i + 1)] + ['{:.5f}'.format(value) for value in result])

    logger.log(99, 'SHAP(tabular batch) completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='SHAP(tabular batch)\n'
                    '\n'
                    'A Unified Approach to Interpreting Model Predictions\n' +
                    'Scott Lundberg, Su-In Lee\n' +
                    'Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017.\n' +
                    'https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html\n' +
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model), default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-i', '--input', help='path to input csv file (csv)', required=True)
    parser.add_argument(
        '-t', '--train', help='path to training dataset csv file (csv)', required=True)
    parser.add_argument(
        '-c2', '--class_index', help='class index (int), default=0', required=True, default=0, type=int)
    parser.add_argument(
        '-a', '--alpha', help='alpha of Ridge, default=0', required=True, default=0, type=float)
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv), default=shap_tabular_batch.csv', required=True, default='shap_tabular_batch.csv')
    parser.set_defaults(func=func)
    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
