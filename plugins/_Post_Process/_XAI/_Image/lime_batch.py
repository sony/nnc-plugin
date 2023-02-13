# Copyright 2023 Sony Group Corporation.
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
from lime_utils.lime_func import lime_batch_func


def main():
    parser = argparse.ArgumentParser(
        description='LIME (all data)\n'
                    '\n'
                    '"Why Should I Trust You?": Explaining the Predictions of Any Classifier\n' +
                    'Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin\n' +
                    'Knowledge Discovery and Data Mining, 2016.\n' +
                    'https://dl.acm.org/doi/abs/10.1145/2939672.2939778\n' +
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input', help='path to input csv file (csv) default=output_result.csv', required=True, default='output_result.csv')
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-v1', '--input_variable', help='Variable to be processed (variable), default=x', required=True, default='x')
    parser.add_argument(
        '-v2', '--label_variable', help='Variable representing class index to visualize (variable) default=y', required=True, default='y')
    parser.add_argument(
        '-n', '--num_samples', help='number of samples N (int), default=1000', required=True, type=int, default=1000)
    parser.add_argument(
        '-s', '--num_segments', help='number of segments (int), default=10', required=True, type=int, default=10)
    parser.add_argument(
        '-s2', '--num_segments_2', help='number of segments to highlight (int), default=3', required=True, type=int, default=3)
    parser.add_argument(
        '-o', '--output', help='path to output image file (image) default=lime.csv', required=True, default='lime.csv')
    parser.set_defaults(func=lime_batch_func)

    args = parser.parse_args()

    file_names = args.func(args)

    save_info_to_csv(args.input, args.output, file_names, column_name='lime')

    logger.log(99, 'LIME (image batch) completed successfully.')


if __name__ == '__main__':
    main()
