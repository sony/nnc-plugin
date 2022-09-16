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
import numpy as np
import matplotlib.pyplot as plt
from nnabla import logger


def plot_fairness(model_fairness, args,
                  ideal_fairness=[0.8, 1.20],
                  metric="Disparate Impact", file_path=None):
    """
    graphical visualization of fairness of the dataset
    Args:
        model_fairness (float) : fairness of the dataset.
        args (dict) : arguments it requires.
        ideal_fairness (list): dataset ideal fairness range.
        metric (str) : name of the fairness metric.
        file_path (str) : location of the file, where the fairness plot to be saved
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharey=True)
    ax.set_ylim([0, max(model_fairness, 1.5)+0.1])
    ax.axhline(y=1.0, color='b', linestyle='-')
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
        ax.text(1.11, 1, "Fair", fontweight='bold', color='Green')
        ax.text(0, model_fairness+0.1,
                str(round(model_fairness, 3)), fontweight='bold', color='Green',
                bbox=dict(facecolor='red', alpha=0.3))
        fig.text(0.92, 0.6, '\n'.join(
            [f"\nFairness :\n{metric} :  {model_fairness:.3f}"]), fontsize='15')
        fig.text(0.92, 0.55, '\n'.join(
            [f"(Dataset is Fair)"]), color='Green', fontweight='bold', fontsize='12')

    else:
        ax.text(0, model_fairness + 0.1,
                str(round(model_fairness, 3)), fontweight='bold', color='Red',
                bbox=dict(facecolor='red', alpha=0.3))
        ax.text(1.11, 1, "Fair", fontweight='bold', color='darkgray')
        fig.text(0.92, 0.6, '\n'.join(
            [f"\nFairness :\n{metric} :  {model_fairness:.3f}"]), fontsize='15')
        if model_fairness < 1:
            fig.text(0.92, 0.55, '\n'.join(
                [f"(Dataset is biased towards privileged group)"]),
                color='red', fontweight='bold', fontsize='12')
            ax.text(1.11, 0.5, "Bias", fontweight='bold', color='red')
        if model_fairness > 1:
            fig.text(0.92, 0.55, '\n'.join(
                [f"(Dataset is biased towards unprivileged group)"]),
                color='red', fontweight='bold', fontsize='12')
            ax.text(1.11, 1.3, "Bias", fontweight='bold', color='red')

    fig.text(0.92, 0.4, '\n'.join(
        [f"Fairness for this metric between {ideal_fairness[0]} and {ideal_fairness[1]}\n"]), fontsize='15')
    fig.suptitle(metric, fontsize=15)
    plt.savefig(file_path, bbox_inches='tight')


def disparate_impact(unprivileged, privileged):
    """
    This metric computed as the ratio of rate of favorable outcome
    for the unprivileged group to that of the privileged group.
    Args:
        unprivileged (numpy.ndarray) : unprivileged class
        privileged (numpy.ndarray) : privileged class
    Returns:
        disparate_impact(float)
    """
    num_positive_unprivileged = np.sum(unprivileged == True)
    num_positive_privileged = np.sum(privileged == True)
    rate_positive_unprivileged = num_positive_unprivileged / \
        unprivileged.shape[0]
    rate_positive_privileged = num_positive_privileged / privileged.shape[0]
    di = rate_positive_unprivileged / rate_positive_privileged
    logger.log(99, 'Disparate impact: %.4f' % (di.item()))
    return round(di.item(), 4)


def func(args):

    logger.log(99, 'Loading dataset ...')
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

    target_col = enum_cols('target_variable', args.target_variable)
    privileged_col = enum_cols('privileged_variable', args.privileged_variable)
    unprivileged_col = enum_cols(
        'unprivileged_variable', args.unprivileged_variable)

    def extract_col(col):
        col_index = in_header.index(col)
        return [float(in_table[i][col_index]) for i in range(len(in_table))]

    y_actual = np.array(extract_col(target_col))
    privileged_variable = np.array(extract_col(privileged_col))
    unprivileged_variable = np.array(extract_col(unprivileged_col))
    # check privileged/unprivileged variables
    if not ((privileged_variable == 1.0) == (unprivileged_variable == 0.0)).all():
        logger.log(
            99, "Both privileged and unprivileged variable values should not be same")
        sys.exit(0)
    out_privileged = y_actual[privileged_variable == 1.0]
    out_unprivileged = y_actual[unprivileged_variable == 1.0]
    metric = "Disparate Impact"
    file_path = os.path.join(os.path.dirname(
        os.path.abspath(args.output)), metric + '.png')
    di = disparate_impact(out_unprivileged, out_privileged)
    plot_fairness(di, args, metric=metric, file_path=file_path)
    with open(args.output, 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Fairness Plot', 'Disparate Impact'])
        writer.writerow([file_path, di])


def main():
    parser = argparse.ArgumentParser(
        description='DisparateImpact\n'
                    'Disparate Impact (Four-Fifths rule/ 80 percent rule),\n'
                    '"This metric computed as the ratio of rate of favorable outcome for '
                    'the unprivileged group to that of the privileged group\n' +
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input',
        help='path to dataset csv file (csv)',
        required=True)
    parser.add_argument(
        '-t', '--target_variable', help='specify the target variable from the input CSV file(variable) ',
        required=True)
    parser.add_argument(
        '-zp', '--privileged_variable',
        help='specify the privileged variable from the input CSV file(variable)',
        required=True)
    parser.add_argument(
        '-zu', '--unprivileged_variable',
        help='specify the unprivileged variable from the input CSV file(variable)',
        required=True)

    parser.add_argument(
        '-n', '--num_samples',
        help='number of samples N (variable), default=all',
        required=True, default='all')
    parser.add_argument(
        '-o', '--output', help='path to output file (csv) default=disparate_impact.csv',
        required=True, default='disparate_impact.csv')
    parser.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)
    logger.log(99, 'completed successfully.')


if __name__ == '__main__':
    main()
