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
import argparse

from nnabla import logger
from utils.file import save_info_to_csv
from shap_utils.shap_func import shap_batch_func


def main():
    parser = argparse.ArgumentParser(
        description='SHAP(batch)\n'
                    '\n'
                    'A Unified Approach to Interpreting Model Predictions\n' +
                    'Scott Lundberg, Su-In Lee\n' +
                    'Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017.\n' +
                    'https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html\n' +
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
    parser.set_defaults(func=shap_batch_func)

    args = parser.parse_args()

    file_names = args.func(args)

    save_info_to_csv(args.input, args.output, file_names, column_name='shap')

    logger.log(99, 'SHAP completed successfully.')


if __name__ == '__main__':
    main()
