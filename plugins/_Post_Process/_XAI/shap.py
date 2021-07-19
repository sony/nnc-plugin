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
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from nnabla.utils.image_utils import imread
from shap_utils.utils import get_executor, red_blue_map, gradient, get_interim_input, plot_shap


def func(args):
    executor = get_executor(args.model)

    # Data source
    data_iterator = (lambda: data_iterator_csv_dataset(
        uri=args.input,
        batch_size=1,
        shuffle=False,
        normalize=not executor.no_image_normalization,
        with_memory_cache=False,
        with_file_cache=False))

    X = imread(args.image)
    if not executor.no_image_normalization:
        X = X / 255.0
    if len(X.shape) == 3:
        X = X.transpose(2, 0, 1)
    else:
        X = X.reshape((1,) + X.shape)

    plot_shap(model=args.model, X=X,
              label=args.class_index, output=args.output,
              interim_layer=args.interim_layer, num_samples=args.num_samples,
              data_iterator=data_iterator, batch_size=args.batch_size,
              red_blue_map=red_blue_map, gradient=gradient, get_interim_input=get_interim_input)

    logger.log(99, "SHAP completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description='SHAP\n'
                    '\n'
                    'A Unified Approach to Interpreting Model Predictions'
                    'Scott Lundberg, Su-In Lee'
                    'https://arxiv.org/abs/1705.07874\n'
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
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
