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

import argparse
from distutils.util import strtobool
from .utils import ensure_dir


def add_finetune_args(parser):
    parser.add_argument(
        "--lmbd",
        "-lm",
        type=float,
        default=0.003,
        required=True,
        help="weight factor of l2, default=0.03",
    )

    parser.add_argument(
        "--epoch", "-ep", type=int, default=1000, required=True, help="finetune epochs, default=1000"
    )

    return parser


def add_feature_extract_args(parser):
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp'
    )

    parser.add_argument(
        "-tr",
        "--input-train",
        help="path to training dataset csv file (csv)",
        required=True,
    )

    parser.add_argument(
        "-v",
        "--input-val",
        help="path to validation dataset csv file (csv)",
        required=True,
    )

    parser.add_argument(
        "--normalize",
        "-n",
        action="store_true",
        default=False,
        help="Image Normaliztion (1.0/255.0) (bool)",
    )
    return parser


def get_basic_args(parser, monitor_path="representer_point_results"):
    parser.add_argument(
        "--shuffle_label",
        type=strtobool,
        default=True,
        help="whether shuffle trainig label or not",
    )

    parser.add_argument("-top", "--top_k", type=int, default=3, required=True,
                        help="top-k influenced samples are presented, default=3")

    parser.add_argument(
        "-N",
        "--num-samples",
        type=int,
        default=10,
        help="Number of test samples for influence analysis, default=10"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path to output csv file (csv) default=influence_sample.csv",
        default="influence_samples.csv",
        required=True,
    )

    parser.add_argument(
        "--device_id",
        "-d",
        type=str,
        default="0",
        required=False,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--type_config",
        "-t",
        type=str,
        default="float",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--context",
        "-c",
        type=str,
        default="cudnn",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--monitor-path", type=str, help=argparse.SUPPRESS, default=monitor_path
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help=argparse.SUPPRESS,
    )
    return parser


def get_infl_args():
    parser = argparse.ArgumentParser(
        description="Representer Point\n"
        + "\n"
        + '"Representer Point Selection for Explaining Deep Neural Networks"\n'
        + "Chih-Kuan Yeh, Joon Sik Kim, Ian E.H. Yen, Pradeep Ravikumar. (2018)\n"
        + "https://arxiv.org/abs/1811.09720\n"
        + "",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser = get_basic_args(parser)
    parser = add_feature_extract_args(parser)
    parser = add_finetune_args(parser)
    args = parser.parse_args()
    ensure_dir(args.monitor_path)
    return args
