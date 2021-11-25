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


def get_basic_args(parser):
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp')
    parser.add_argument(
        '-bs', '--batch_size', help='batch size for train, infl default=32', default=32, type=int)
    # epoch for SGD influence calculation(int). It becomes 1 when model nnp is
    # designated.
    parser.add_argument(
        '-e', '--n_epochs', help=argparse.SUPPRESS, default=20, type=int)
    # device id of gpu
    parser.add_argument(
        '-di', '--device-id', help=argparse.SUPPRESS, default=0)
    return parser


def get_train_infl_args():
    parser = argparse.ArgumentParser(
        description='SGD Influence (image)\n' +
        '\n' +
        '"Data Cleansing for Models Trained with SGD"\n' +
        '  Satoshi Hara, Atsushi Nitanda, and Takanori Maehara (2019)\n' +
        'https://papers.nips.cc/paper/8674-data-cleansing-for-models-trained-with-sgd\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-t', '--input-train', help='path to training dataset csv file (csv)', required=True)
    parser.add_argument(
        '-v', '--input-val', help='path to validation dataset csv file (csv)', required=True)
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv) default=influence.csv', default='influence.csv')
    parser.add_argument(
        '-nt', '--n_trials', help='number or trials default=6', default=6, type=int)
    # path to score csv file
    parser.add_argument(
        '-so', '--score_output', help=argparse.SUPPRESS, default=None)
    # save dir name default=sgd_infl_results
    parser.add_argument(
        '-d', '--weight_output', help=argparse.SUPPRESS, default='sgd_infl_results', type=str)
    # if True, save all params with which sgd-influence is calculated.
    # Otherwise, save and use only final model.
    parser.add_argument(
        '-a', '--calc-infl-with-all-params', help=argparse.SUPPRESS, action='store_true')
    # calc method of sgd-influence default=last
    parser.add_argument(
        '-c', '--calc-infl-method', help=argparse.SUPPRESS, default='last', choices=['last', 'all'])
    parser = get_basic_args(parser)

    return parser.parse_args()


def get_train_infl_args_of_inflence_functions():
    parser = argparse.ArgumentParser(
        description='Influence Functions (image)\n' +
        '\n' +
        '"Understanding Black-box Predictions via Influence Functions"\n' +
        '  Pang Wei Koh, Percy Liang.\n' +
        'https://arxiv.org/abs/1703.04730\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-t', '--input-train', help='path to training dataset csv file (csv)', required=True)
    parser.add_argument(
        '-v', '--input-val', help='path to validation dataset csv file (csv)', required=True)
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv) default=influence(influence_functions).csv', default='influence(influence_functions).csv')
    parser.add_argument(
        '-nt', '--n_trials', help='number or trials default=6', default=6, type=int)
    # path to score csv file
    parser.add_argument(
        '-so', '--score_output', help=argparse.SUPPRESS, default=None)
    # save dir name default=sgd_infl_results
    parser.add_argument(
        '-d', '--weight_output', help=argparse.SUPPRESS, default='influence_functions_results', type=str)
    # calc method of sgd-influence default=last
    parser.add_argument(
        '-c', '--calc-infl-method', help=argparse.SUPPRESS, default='last', choices=['last', 'all'])
    # alpha for loss calculation
    parser.add_argument(
        '-ap', '--alpha', help=argparse.SUPPRESS, default=0.1, type=float)
    # if True, save all params with which sgd-influence is calculated. Otherwise, save and use only final model.
    parser.add_argument(
        '-a', '--calc-infl-with-all-params', help=argparse.SUPPRESS, default=True, type=bool)
    parser = get_basic_args(parser)

    return parser.parse_args()


def get_eval_args():
    parser = argparse.ArgumentParser(
        description='check performance of SGD-influence', formatter_class=argparse.RawTextHelpFormatter)
    parser = get_basic_args(parser)
    parser.add_argument(
        '-o', '--output_dir', help='path to output dir', required=True)
    parser.add_argument(
        '-nt', '--n_trials', help='number or trials ', default=6, type=int)
    parser.add_argument(
        '-r', '--remove_n_list', help="list of n of samples to remove. ex: '-r 10 20' makes [10, 20]",
        type=int, nargs='+', default=[0, 1, 10, 40, 100, 200, 600, 1000, 3000, 5000, 10000, 30000])
    parser.add_argument(
        '-ds', '--dataset', choices=['cifar10', 'stl10', 'mnist'], default='cifar10')
    parser.add_argument(
        '-ra', '--retrain_all', help='if True, retrain all.', action='store_true')
    return parser.parse_args()
