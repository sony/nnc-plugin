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
import csv
import os
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from nnabla import logger


def plot_fairness(model_fairness, args,
                  ideal_fairness=[-0.10, 0.10],
                  metric="Demographic Parity", file_path=None):
    """
    graphical visualization of fairness of the model
    Args:
        model_fairness (float): fairness of the model.
        args (dict) : arguments it requires.
        ideal_fairness (list) : model ideal fairness range.
        metric (str): name of the fairness metric.
        file_path (str) : location of the file, where the fairness plot to be saved
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharey=True)
    ax.set_ylim([-1, 1])
    ax.axhline(y=0.0, color='b', linestyle='-')
    ax.bar(['Fairness'], [model_fairness], color="darkcyan", width=2)
    ax.axhspan(ideal_fairness[0], ideal_fairness[1],
               facecolor='0.5', color="lightgreen", alpha=0.5)
    ax.set_ylabel(metric)

    fig.text(0.92, 0.7, '\n'.join(
        [f"Privileged Variable : {args.privileged_variable} \n"
         f"Unprivileged Variable : {args.unprivileged_variable} \n"
         f"Total number of samples : {args.num_samples} \n"]),
        fontsize='15')

    if ideal_fairness[0] <= model_fairness <= ideal_fairness[1]:
        ax.text(1.11, 0, "Fair", fontweight='bold', color='Green')
        fig.text(0.92, 0.6, '\n'.join(
            [f"\nFairness :\n{metric} :  {model_fairness:.3f}"]), fontsize='15')
        fig.text(0.92, 0.55, '\n'.join(
            [f"(Model is Fair)"]), color='Green', fontweight='bold', fontsize='12')
        if model_fairness < 0:
            ax.text(0, model_fairness - 0.1,
                    str(round(model_fairness, 3)), fontweight='bold', color='Green',
                    bbox=dict(facecolor='red', alpha=0.3))
            ax.text(1.11, -0.5, "Bias", fontweight='bold', color='darkgray')
        else:
            ax.text(0, model_fairness + 0.1,
                    str(round(model_fairness, 3)), fontweight='bold', color='Green',
                    bbox=dict(facecolor='red', alpha=0.3))
            ax.text(1.11, 0.5, "Bias", fontweight='bold', color='darkgray')
    else:
        ax.text(1.11, 0, "Fair", fontweight='bold', color='darkgray')
        fig.text(0.92, 0.6, '\n'.join(
            [f"\nFairness :\n{metric} :  {model_fairness:.3f}"]), fontsize='15')
        if model_fairness < 0:
            ax.text(0, model_fairness - 0.1,
                    str(round(model_fairness, 3)), fontweight='bold', color='red',
                    bbox=dict(facecolor='red', alpha=0.3))
            ax.text(1.11, -0.5, "Bias", fontweight='bold', color='red')
            fig.text(0.92, 0.55, '\n'.join(
                [f"(Model is biased towards privileged group)"]),
                color='red', fontweight='bold', fontsize='12')
        else:
            ax.text(0, model_fairness + 0.1,
                    str(round(model_fairness, 3)), fontweight='bold', color='red',
                    bbox=dict(facecolor='red', alpha=0.3))
            ax.text(1.11, 0.5, "Bias", fontweight='bold', color='red')
            fig.text(0.92, 0.55, '\n'.join(
                [f"(Model is biased towards unprivileged group)"]),
                color='red', fontweight='bold', fontsize='12')

    fig.text(0.92, 0.4, '\n'.join(
        [f"Fairness for this metric between {ideal_fairness[0]} and {ideal_fairness[1]}\n"]), fontsize='15')

    fig.suptitle(metric, fontsize=15)
    plt.savefig(file_path, bbox_inches='tight')


def demographic_parity(output_unprivileged, output_privileged, classification_threshold):
    """
    Computed as the difference between the rate of positive outcomes
    received by the unprivileged group to the privileged group.
    Args:
        output_unprivileged (numpy.ndarray) : output of unprivileged class
        output_privileged (numpy.ndarray) : output of privileged class
        classification_threshold (float) : classification threshold
    Returns:
        Demographic parity (float)
    """
    unprivileged_pred = (output_unprivileged >= classification_threshold)
    privileged_pred = (output_privileged >= classification_threshold)
    num_positive_unprivileged = np.sum(unprivileged_pred == True)
    num_positive_privileged = np.sum(privileged_pred == True)
    rate_positive_unprivileged = num_positive_unprivileged / \
        unprivileged_pred.shape[0]
    rate_positive_privileged = num_positive_privileged / \
        privileged_pred.shape[0]
    dpd = rate_positive_unprivileged - rate_positive_privileged
    logger.log(99, 'Demographic parity: %.4f' % (dpd.item()))
    return round(dpd.item(), 4)


def func(args):

    logger.log(99, 'Loading Evaluated dataset ...')
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        in_header = next(reader)
        in_table = [[col.strip() for col in row]
                    for row in reader if len(row) > 0]
    if (args.num_samples.lower().strip() == 'all') or (int(args.num_samples) > len(in_table)):
        in_table = in_table
    else:
        in_table = in_table[:int(args.num_samples)]
    args.num_samples = len(in_table)

    def enum_cols(name, col):
        if col not in in_header:
            logger.critical(
                'The column named {} specified by {} is not found in the input dataset'.format(
                    col, name))
        else:
            logger.log(99, '{} : {}'.format(col, name))
        return col

    output_col = enum_cols('output-variable', args.output_variable)
    privileged_col = enum_cols('privileged_variable', args.privileged_variable)
    unprivileged_col = enum_cols(
        'unprivileged_variable', args.unprivileged_variable)

    def extract_col(col):
        col_index = in_header.index(col)
        return [float(in_table[i][col_index]) for i in range(len(in_table))]

    clf_out = np.asarray(extract_col(output_col)).flatten()
    privileged_variable = np.array(extract_col(privileged_col))
    unprivileged_variable = np.array(extract_col(unprivileged_col))
    # check privileged / unprivileged variables
    if not ((privileged_variable == 1.0) == (unprivileged_variable == 0.0)).all():
        logger.log(
            99, "Both privileged and unprivileged variable values should not be same")
        sys.exit(0)

    out_privileged = clf_out[privileged_variable == 1.0]
    out_unprivileged = clf_out[unprivileged_variable == 1.0]
    fair_flag = abs(args.fair_threshold) <= 1  # check the fairness value
    if not fair_flag:
        logger.error(f'Fairness value out of range! '
                     f'Fairness of this metric is between -1.0 and 1.0 , Please enter again', exc_info=0)
        sys.exit(0)
    else:
        logger.log(99,
                   f'Fairness of this metric is between {-abs(args.fair_threshold)} and {-abs(args.fair_threshold)}')
        ideal_fairness = [-abs(args.fair_threshold), abs(args.fair_threshold)]
        metric = "Demographic Parity"
        file_path = os.path.join(os.path.dirname(
            os.path.abspath(args.output)), metric + '.png')

    dpd = demographic_parity(
        out_unprivileged, out_privileged, args.clf_threshold)
    plot_fairness(dpd, args, metric=metric,
                  ideal_fairness=ideal_fairness, file_path=file_path)
    with open(args.output, 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Fairness Plot', 'Demographic Parity'])
        writer.writerow([file_path, dpd])


def main():
    parser = argparse.ArgumentParser(
        description='Demographic Parity\n'
                    '\n'
                    'Demographic/Statistical Parity,\n' +
                    'This metric is computed as the difference between the rate of '
                    'positive outcomes in unprivileged and privileged groups\n' +
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input',
        help='path to input csv file, output result shown in the evaluation tab (csv) default=output_result.csv',
        required=True, default='output_result.csv')
    parser.add_argument(
        '-t', '--output_variable',
        help="specify the target output variable, expect sigmoid output (variable) default=y'",
        required=True, default="y'")
    parser.add_argument(
        '-zp', '--privileged_variable',
        help='specify the privileged variable from the input CSV file (variable)',
        required=True)
    parser.add_argument(
        '-zu', '--unprivileged_variable',
        help='specify the unprivileged variable from the input CSV file (variable)',
        required=True)
    parser.add_argument(
        '-th', '--clf_threshold',
        help='specify best optimal classification threshold, default=0.5',
        required=True, type=float, default=0.5)
    parser.add_argument(
        '-fair_th',
        '--fair_threshold',
        help='Specify fairness threshold, default value is 0.10, default=0.10',
        type=float, default=0.10)
    parser.add_argument(
        '-n', '--num_samples',
        help='number of samples N (variable), default=all',
        required=True, default='all')
    parser.add_argument(
        '-o', '--output', help='path to output file (csv) default=demographic_parity.csv',
        required=True, default='demographic_parity.csv')
    parser.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)
    logger.log(99, 'completed successfully.')


if __name__ == '__main__':
    main()
