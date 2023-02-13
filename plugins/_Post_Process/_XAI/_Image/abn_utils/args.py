
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


def add_basic_args(parser):
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model), default=results.nnp', default='results.nnp')
    parser.add_argument(
        '-aml', '--attention-map-layer', help='name of attention map layer, default=Attention_Map', default='Attention_Map')
    return parser


def add_single_image_args(parser):
    parser.add_argument(
        '-i', '--image', help='path to input image file (image)', required=True)
    parser.add_argument(
        '-o', '--output', help='path to output image file (image) default=abn_attention_map.png', default='abn_attention_map.png')
    return parser


def add_multi_image_args(parser):
    parser.add_argument(
        '-i', '--input', help='path to input csv file (csv) default=output_result.csv', required=True, default='output_result.csv')
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv) default=abn_attention_map.csv', default='abn_attention_map.csv')
    return parser


def get_single_image_args(parser):
    parser = add_basic_args(parser)
    parser = add_single_image_args(parser)
    return parser.parse_args()


def get_multi_image_args(parser):
    parser = add_basic_args(parser)
    parser = add_multi_image_args(parser)
    return parser.parse_args()
