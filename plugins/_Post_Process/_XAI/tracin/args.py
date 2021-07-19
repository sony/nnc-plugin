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
from tracin.utils import ensure_dir
from distutils.util import strtobool


def bandwidth_limit(x, minimum=30):
    x = int(x)
    if x < minimum:
        raise argparse.ArgumentTypeError(f"Minimum bandwidth is {minimum}")
    return x


def add_train_args(parser):
    parser.add_argument('-t',
                        '--input-train',
                        help='path to training dataset csv file (csv)',
                        required=True)
    # path to validation dataset csv file (csv)
    parser.add_argument('-v',
                        '--input-val',
                        help=argparse.SUPPRESS,
                        required=False)
    return parser


def add_infl_args(parser):
    # whether save influence score between every epoch or not
    parser.add_argument('--save_every_epoch',
                        type=strtobool,
                        default=False,
                        help=argparse.SUPPRESS)
    return parser


def add_basic_args(parser, monitor_path='tracin_infl_results'):
    model_save_path = monitor_path

    parser.add_argument('--monitor-path', '-m', type=str,
                        help=argparse.SUPPRESS, default=monitor_path)
    # Directory to save checkpoints and logs.
    parser.add_argument('--model_save_path',
                        type=str,
                        default=model_save_path,
                        help=argparse.SUPPRESS)

    parser.add_argument('--model_save_interval',
                        type=input,
                        help=argparse.SUPPRESS,
                        default=1)
    # whether shuffle trainig label or not
    parser.add_argument('--shuffle_label',
                        type=strtobool,
                        default=False,
                        help=argparse.SUPPRESS)

    parser.add_argument('--augmentation', type=strtobool,
                        default=True, help=argparse.SUPPRESS)

    parser.add_argument('--model',
                        '-md',
                        type=str,
                        default='resnet23',
                        help='model to use calc influence (resnet23 or resnet56) default=resnet23')
    # directory to read saved data and weight
    parser.add_argument('--weight_input',
                        type=str,
                        help=argparse.SUPPRESS,
                        default=model_save_path)

    parser.add_argument('-o', '--output',
                        help='path to output csv file (csv) default=tracin_self_influence.csv',
                        default='tracin_self_influence.csv')
    # directory to save data and weight
    parser.add_argument('--weight_output',
                        type=str,
                        help=argparse.SUPPRESS,
                        default=model_save_path)
    # path to save checkpoint
    parser.add_argument('--checkpoint',
                        type=str,
                        default=model_save_path,
                        help=argparse.SUPPRESS)
    # resume epoch
    parser.add_argument('--resume', type=int, default=0,
                        help=argparse.SUPPRESS)
    # Device ID the training run on when gpu is available
    parser.add_argument(
        "--device_id",
        "-d",
        type=str,
        default='0',
        help=argparse.SUPPRESS
    )
    # Type of computation. e.g. "float", "half".
    parser.add_argument("--type_config",
                        type=str,
                        default='float',
                        help=argparse.SUPPRESS)

    parser.add_argument('--train_batch_size',
                        '-tbs',
                        type=int,
                        default=128,
                        help='Batch size for training (per replica). default=128')

    parser.add_argument('--val-iter', type=int,
                        default=100, help=argparse.SUPPRESS)
    # Batch size for eval. (per replica)
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=128,
                        help=argparse.SUPPRESS)
    parser.add_argument('--train_epochs',
                        '-te',
                        type=bandwidth_limit,
                        default=100,
                        help='Number of epochs to train for. default=100')
    # Number of epochs of warmup.
    parser.add_argument('--warmup_epochs',
                        type=int,
                        default=15,
                        help=argparse.SUPPRESS)
    # Boundaries for learning rate decay.
    parser.add_argument('--boundaries',
                        nargs='*',
                        default=[15, 90, 180, 240],
                        help=argparse.SUPPRESS)
    # Multipliers for learning rate decay.
    parser.add_argument('--multipliers',
                        nargs='*',
                        default=[1.0, 0.1, 0.01, 0.001],
                        help=argparse.SUPPRESS)
    parser.add_argument('--seed',
                        '-s',
                        help='random seed number default=0',
                        default=0,
                        type=int)
    return parser


def get_infl_args():
    parser = argparse.ArgumentParser()
    parser = add_basic_args(parser)
    parser = add_infl_args(parser)
    args = parser.parse_args()
    return args


def get_train_args():
    parser = argparse.ArgumentParser(
        description='Training TrackIn Base model')
    parser = add_basic_args(parser)
    parser = add_train_args(parser)
    args = parser.parse_args()
    ensure_dir(args.model_save_path)
    return args


def get_train_infl_args():
    parser = argparse.ArgumentParser(
        description='TracIn\n' +
        '\n' +
        '"Estimating Training Data Influence by Tracing Gradient Descent"\n' +
        'Garima Pruthi, Frederick Liu, Mukund Sundararajan and Satyen Kale (2020)\n' +
        'https://arxiv.org/abs/2002.08484\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser = add_train_args(parser)
    parser = add_basic_args(parser)
    parser = add_infl_args(parser)
    args = parser.parse_args()
    ensure_dir(args.model_save_path)
    return args
