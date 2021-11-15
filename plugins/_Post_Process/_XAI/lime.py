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
from nnabla.utils.image_utils import imsave
from lime_utils.lime_func import lime_func


def main():
    parser = argparse.ArgumentParser(
        description='LIME (image)\n' +
        '\n' +
        '"Why Should I Trust You?": Explaining the Predictions of Any Classifier' +
        'Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin' +
        'https://arxiv.org/abs/1602.04938\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-i', '--image', help='path to input image file (image)', required=True)
    parser.add_argument(
        '-c', '--class_index', help='class index to visualize (int), default=0', required=True, type=int, default=0)
    parser.add_argument(
        '-n', '--num_samples', help='number of samples N (int), default=1000', required=True, type=int, default=1000)
    parser.add_argument(
        '-s', '--num_segments', help='number of segments (int), default=10', required=True, type=int, default=10)
    parser.add_argument(
        '-s2', '--num_segments_2', help='number of segments to highlight (int), default=3', required=True, type=int, default=3)
    parser.add_argument(
        '-o', '--output', help='path to output image file (image) default=lime.png', required=True, default='lime.png')
    parser.set_defaults(func=lime_func)

    args = parser.parse_args()

    result = args.func(args)
    imsave(args.output, result)
    logger.log(99, 'LIME (image) completed successfully.')


if __name__ == '__main__':
    main()
