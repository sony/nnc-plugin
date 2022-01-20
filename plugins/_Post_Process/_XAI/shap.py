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
from nnabla import logger
from shap_utils.shap_func import shap_func


def main():
    parser = argparse.ArgumentParser(
        description='SHAP\n'
                    '\n'
                    'A Unified Approach to Interpreting Model Predictions\n' +
                    'Scott Lundberg, Su-In Lee\n' +
                    'Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017.\n' +
                    'https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html\n' +
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--image', help='path to input image file (image)', required=True)
    parser.add_argument(
        '-in', '--input', help='path to input csv file (csv) default=output_result.csv', required=True, default='output_result.csv')
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-c', '--class_index', help='class index to visualize (int), default=0', required=True, type=int, default=0)
    parser.add_argument(
        '-n', '--num_samples', help='number of samples N (int), default=100', required=True, type=int, default=100)
    parser.add_argument(
        '-b', '--batch_size', help=' batch size, default=50', required=True, type=int, default=50)
    parser.add_argument(
        '-il', '--interim_layer', help='layer input to explain, default=0', required=True, type=int, default=0)
    parser.add_argument(
        '-o', '--output', help='path to output image file (image) default=shap.png', required=True, default='shap.png')
    parser.set_defaults(func=shap_func)

    args = parser.parse_args()

    args.func(args)
    logger.log(99, "SHAP completed successfully")


if __name__ == '__main__':
    main()
