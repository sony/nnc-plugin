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
from nnabla import logger
import matplotlib.pyplot as plt


def plot_fairness(model_fairness, args,
                  ideal_fairness=[0, 0.10],
                  metric="Theil index", file_path=None):
    """
    graphical visualization of fairness of the model
    Args:
        model_fairness (float) : fairness of the ML model.
        args (dict) : arguments it requires.
        ideal_fairness (list): ideal model fairness range.
        metric (str) : name of the fairness metric.
        file_path (str) : location of the file, where the fairness plot to be saved
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharey=True)
    ax.set_ylim([0.0, 1])
    ax.axhline(y=0.0, color='b', linestyle='-')
    ax.bar(['Fairness'], [model_fairness], color="darkcyan", width=2)
    ax.axhspan(ideal_fairness[0], ideal_fairness[1],
               facecolor='0.5', color="lightgreen", alpha=0.5)
    ax.set_ylabel(metric)
    fig.text(0.92, 0.8, '\n'.join(
        [f"Total number of samples : {args.num_samples} \n"]),
        fontsize='15')
    if ideal_fairness[0] < model_fairness < ideal_fairness[1]:
        ax.text(1.11, 0, "Fair", fontweight='bold', color='Green')
        ax.text(1.11, 0.5, "Bias", fontweight='bold', color='darkgray')
        ax.text(0, model_fairness + 0.05,
                str(round(model_fairness, 3)), fontweight='bold',
                color='Green', bbox=dict(facecolor='red', alpha=0.3))
        fig.text(0.92, 0.6, '\n'.join(
            [f"\nFairness :\n{metric} :  {model_fairness:.3f}"]), fontsize='15')
        fig.text(0.92, 0.55, '\n'.join(
            [f"(Model is Fair)"]), color='Green', fontweight='bold', fontsize='12')
    else:
        ax.text(1.11, 0, "Fair", fontweight='bold', color='darkgray')
        ax.text(1.11, 0.5, "Bias", fontweight='bold', color='red')
        ax.text(0, model_fairness + 0.05,
                str(round(model_fairness, 3)), fontweight='bold',
                color='red', bbox=dict(facecolor='red', alpha=0.3))
        fig.text(0.92, 0.6, '\n'.join(
            [f"\nFairness :\n{metric} :  {model_fairness:.3f}"]), fontsize='15')
        fig.text(0.92, 0.55, '\n'.join(
            [f"(Model is Biased)"]), color='red', fontweight='bold', fontsize='12')

    fig.text(0.92, 0.4, '\n'.join(
        [f"Fairness for this metric between {ideal_fairness[0]} and {ideal_fairness[1]}\n"]), fontsize='15')
    fig.suptitle(metric, fontsize=15)
    plt.savefig(file_path, bbox_inches='tight')


def generalized_entropy_index(y_actual, y_pred, classification_threshold, alpha=1):
    """
    Generalized entropy index measures inequality over a population
    Args:
        y_actual (numpy.ndarray) : true data (or target, ground truth)
        y_pred (numpy.ndarray) : predicted data (classifier output)
        alpha (int) : parameter that regulates the weight given to distances between values at
                      different parts of the distribution,
                      value of 0 is equivalent to the mean log deviation,
                      1 is the Theil index,
                      and 2 is half the squared coefficient of variation.
        classification_threshold (float) : classification threshold
    Returns:
        gei :generalized entropy index (float)

    References:
            Speicher, Till, Hoda Heidari, Nina Grgic-Hlaca, Krishna P. Gummadi, Adish Singla,
            Adrian Weller, and Muhammad Bilal Zafar. "A unified approach to quantifying algorithmic
            unfairness: Measuring individual &group unfairness via inequality indices."
            In Proceedings of the 24th ACM SIGKDD international conference on knowledge
            discovery & data mining, pp. 2239-2248. 2018.
    """
    y_pred = np.where(y_pred > classification_threshold, 1, 0)
    generalized_entropy_benefit = 1 + y_pred - \
        y_actual  # compute the benefit for individuals

    if alpha == 1:
        gei = np.mean(np.log((generalized_entropy_benefit / np.mean(generalized_entropy_benefit))
                      ** generalized_entropy_benefit) / np.mean(generalized_entropy_benefit))
    elif alpha == 0:
        gei = -np.mean(np.log(generalized_entropy_benefit / np.mean(
            generalized_entropy_benefit)) / np.mean(generalized_entropy_benefit))
    else:
        gei = np.mean((generalized_entropy_benefit / np.mean(generalized_entropy_benefit))
                      ** alpha - 1) / (alpha * (alpha - 1))

    logger.log(99, 'generalized_entropy_index: %.4f' % (gei.item()))
    return round(gei.item(), 4)


def func(args):

    logger.log(99, 'Loading Evaluated Results ...')
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

    actual_label_col = enum_cols('target_variable', args.target_variable)
    output_col = enum_cols('output-variable', args.output_variable)

    def extract_col(col):
        col_index = in_header.index(col)
        return [float(in_table[i][col_index]) for i in range(len(in_table))]

    y_actual = np.array(extract_col(actual_label_col))
    clf_out = np.asarray(extract_col(output_col)).flatten()
    fair_flag = abs(args.fair_threshold) <= 1  # check the fairness value
    if not fair_flag:
        logger.error(f'Fairness value out of range! '
                     f'Fairness of this metric is between 0.0 and 1.0 , Please enter again', exc_info=0)
        sys.exit(0)
    else:
        logger.log(99,
                   f'Fairness for this metric is between 0 and {-abs(args.fair_threshold)}')
        ideal_fairness = [0, abs(args.fair_threshold)]
        metric = "Theil Index"
        file_path = os.path.join(os.path.dirname(
            os.path.abspath(args.output)), metric + '.png')

    ti = generalized_entropy_index(
        y_actual, clf_out, args.clf_threshold, alpha=1)  # alpha =1 for theil index
    plot_fairness(ti, args, metric=metric,
                  ideal_fairness=ideal_fairness, file_path=file_path)
    with open(args.output, 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Fairness Plot', 'Theil Index'])
        writer.writerow([file_path, ti])


def main():
    parser = argparse.ArgumentParser(
        description='Theil Index\n'
                    '\n'
                    'Theil Index,\n' +
                    'Computed as the generalized entropy of benefit for '
                    'all individuals in the dataset,\n'
                    'With alpha = 1. It measures the inequality in '
                    'benefit allocation for individuals . \n' +
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input',
        help='path to input csv file, output result shown in the '
             'evaluation tab (csv) default=output_result.csv',
        required=True, default='output_result.csv')

    parser.add_argument(
        '-t', '--target_variable',
        help='specify the actual target(label) variable (variable)',
        required=True)
    parser.add_argument(
        '-p', '--output_variable',
        help="specify the predicted output variable, "
             "expect sigmoid output (variable) default=y'",
        required=True, default="y'")

    parser.add_argument(
        '-th', '--clf_threshold',
        help='specify optimal classification threshold, default=0.5',
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
        '-o', '--output', help='path to output file (csv) default=theil_index.csv',
        required=True, default='theil_index.csv')
    parser.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)
    logger.log(99, 'completed successfully.')


if __name__ == '__main__':
    main()
