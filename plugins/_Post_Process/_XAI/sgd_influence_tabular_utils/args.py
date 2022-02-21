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


def add_basic_args(parser):
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp')
    parser.add_argument(
        '-bs', '--batch_size', help='batch size for train, infl default=64', default=64, type=int)
    # epoch for SGD influence calculation(int). It becomes 1 when model nnp is designated.
    parser.add_argument(
        '-e', '--num-epochs', help=argparse.SUPPRESS, default=20, type=int)
    # device id of gpu
    parser.add_argument(
        '-di', '--device-id', help=argparse.SUPPRESS, default=0)
    # if True, retrain all.
    parser.add_argument(
        '-ra', '--retrain_all', help=argparse.SUPPRESS, action='store_true')
    # if True, save and use only final model, Otherwise, save all params with which sgd-influence is calculated.
    parser.add_argument(
        '-cl', '--only-last-params', help=argparse.SUPPRESS, action='store_true')
    # target data for train_infl_eval
    parser.add_argument(
        '-tg', '--target', help=argparse.SUPPRESS, default='adult', type=str)
    # temporary directory default=sgd_tabular_result
    parser.add_argument(
        '-td', '--temp-dir', help=argparse.SUPPRESS, type=str, default='sgd_tabular_result')
    # Variable representing class index to visualize (variable) default=y
    parser.add_argument(
        '-lv', '--label_variable', help=argparse.SUPPRESS, default='y', type=str)
    return parser


def add_train_infl_args(parser):
    parser.add_argument(
        '-t', '--input-train', help='path to training dataset csv file (csv)', required=True)
    parser.add_argument(
        '-v', '--input-val', help='path to validation dataset csv file (csv)', required=True)
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv) default=sgd_influence_tabular.csv', default='sgd_influence_tabular.csv')
    return parser


def add_eval_args(parser):
    parser.add_argument(
        '-o', '--output_dir', help='path to output dir', required=True)
    parser.add_argument(
        '-nt', '--n_trials', help='number or trials ', default=6, type=int)
    parser.add_argument(
        '-r', '--remove_n_list', help="list of n of samples to remove. ex: '-r 10 20' makes [10, 20]",
        type=int, nargs='+', default=[0, 1, 10, 40, 100, 200, 600, 1000, 3000, 5000, 10000, 30000])
    parser.add_argument(
        '-ds', '--dataset', choices=['adult', 'iris', 'premium'], default='adult')
    return parser


def get_train_args():
    parser = argparse.ArgumentParser(description='Train Models & Save')
    parser = add_basic_args(parser)
    parser = add_train_infl_args(parser)
    return parser.parse_args()


def get_infl_args():
    parser = argparse.ArgumentParser(description='Train Models & Save')
    parser = add_basic_args(parser)
    parser = add_train_infl_args(parser)
    return parser.parse_args()


def get_eval_args():
    parser = argparse.ArgumentParser(
        description='check performance of SGD-influence', formatter_class=argparse.RawTextHelpFormatter)
    parser = add_basic_args(parser)
    parser = add_eval_args(parser)
    return parser.parse_args()


def get_train_infl_args():
    parser = argparse.ArgumentParser(
        description='SGD Influence (tabular)\n' +
        '\n' +
        '"Data Cleansing for Models Trained with SGD"\n' +
        '  Satoshi Hara, Atsushi Nitanda, and Takanori Maehara (2019)\n' +
        'https://papers.nips.cc/paper/8674-data-cleansing-for-models-trained-with-sgd\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-nt', '--n_trials', help='number or trials ', default=6, type=int)
    parser = add_basic_args(parser)
    parser = add_train_infl_args(parser)
    return parser.parse_args()


def get_train_infl_eval_args():
    return get_eval_args()
