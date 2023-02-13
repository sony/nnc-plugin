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


def add_basic_args(parser):
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-nl', '--noise_level', help='noise level(0.0 to 1.0) to calculate standard deviation for input image, default=0.15', type=float, default=0.15)
    parser.add_argument(
        '-n', '--num_samples', help='number of samples to average smoothgrad results, default=25', type=int, default=25)
    # index of layer of interest to visualize (input layer is 0), default=0
    parser.add_argument(
        '-li', '--layer_index', help=argparse.SUPPRESS, type=int, default=0)
    return parser


def add_single_image_args(parser):
    parser.add_argument(
        '-i', '--image', help='path to input image file (image)', required=True)
    parser.add_argument(
        '-c', '--class_index', help='class index to visualize (int), default=0', required=True, type=int, default=0)
    parser.add_argument(
        '-o', '--output', help='path to output image file (image) default=smoothgrad.png', required=True, default='smoothgrad.png')
    return parser


def add_multi_image_args(parser):
    parser.add_argument(
        '-i', '--input', help='path to input csv file (csv) default=output_result.csv', required=True, default='output_result.csv')
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv) default=smoothgrad.csv', required=True, default='smoothgrad.csv')
    parser.add_argument(
        '-v1', '--input_variable', help='Variable to be processed (variable), default=x', required=True, default='x')
    parser.add_argument(
        '-v2', '--label_variable', help='Variable representing class index to visualize (variable) default=y', required=True, default='y')
    return parser


def get_single_image_args(parser):
    parser = add_basic_args(parser)
    parser = add_single_image_args(parser)
    return parser.parse_args()


def get_multi_image_args(parser):
    parser = add_basic_args(parser)
    parser = add_multi_image_args(parser)
    return parser.parse_args()
