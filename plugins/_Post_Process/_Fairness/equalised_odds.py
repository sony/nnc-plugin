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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nnabla import logger


def plot_fairness(model_fairness, args,
                  ideal_fairness=[0, 0.10],
                  metric="Equalised Odds", file_path=None):
    """
    graphical visualization of fairness of the model
    Args:
        model_fairness (float) : fairness of the model.
        args (dict): arguments it requires.
        ideal_fairness (list) : model ideal fairness range.
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
    fig.text(0.92, 0.7, '\n'.join(
        [f"Privileged Variable : {args.privileged_variable} \n"
         f"Unprivileged Variable : {args.unprivileged_variable} \n"
         f"Total number of samples : {args.num_samples} \n"]),
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


def average_abs_equalised_odd(y_privileged, y_unprivileged, preds_privileged,
                              preds_unprivileged, classification_threshold):
    """
    Computed as average of absolute difference between false positive rate and true positive rate
    for unprivileged and privileged groups.
    Args:
        y_privileged (numpy.ndarray) : actual privileged class
        y_unprivileged (numpy.ndarray) : actual un privileged class
        preds_privileged (numpy.ndarray) : predicted output of privileged class
        preds_unprivileged (numpy.ndarray) : predicted output of unprivileged class
        classification_threshold (float) : classification threshold
    Returns:
        average absolute equalised odds (float)
    """
    privileged_pred = (preds_privileged >= classification_threshold)
    unprivileged_pred = (preds_unprivileged >= classification_threshold)
    cm_privileged = confusion_matrix(y_privileged, privileged_pred)
    cm_unprivileged = confusion_matrix(y_unprivileged, unprivileged_pred)
    unprivileged_TPR = cm_unprivileged[1, 1] / cm_unprivileged[1].sum()
    privileged_TPR = cm_privileged[1, 1] / cm_privileged[1].sum()
    unprivileged_FPR = cm_unprivileged[0, 1] / cm_unprivileged[0].sum()
    privileged_FPR = cm_privileged[0, 1] / cm_privileged[0].sum()
    average_abs_odds_difference = 0.5 * (
        abs(unprivileged_FPR - privileged_FPR) + abs(unprivileged_TPR - privileged_TPR))
    logger.log(99, 'average absolute equalised odds: %.4f' %
               (average_abs_odds_difference.item()))
    return round(average_abs_odds_difference.item(), 4)


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
    privileged_col = enum_cols('privileged_variable', args.privileged_variable)
    unprivileged_col = enum_cols(
        'unprivileged_variable', args.unprivileged_variable)

    def extract_col(col):
        col_index = in_header.index(col)
        return [float(in_table[i][col_index]) for i in range(len(in_table))]

    y_actual = np.array(extract_col(actual_label_col))
    clf_out = np.asarray(extract_col(output_col)).flatten()
    privileged_variable = np.array(extract_col(privileged_col))
    unprivileged_variable = np.array(extract_col(unprivileged_col))
    # check privileged/unprivileged variables
    if not ((privileged_variable == 1.0) == (unprivileged_variable == 0.0)).all():
        logger.log(
            99, "Both privileged and unprivileged variable values should not be same")
        sys.exit(0)
    y_privileged = y_actual[privileged_variable == 1.0]
    y_unprivileged = y_actual[unprivileged_variable == 1.0]
    preds_privileged = clf_out[privileged_variable == 1.0]
    preds_unprivileged = clf_out[unprivileged_variable == 1.0]
    fair_flag = abs(args.fair_threshold) <= 1  # check the fairness value
    if not fair_flag:
        logger.error(f'Fairness value out of range! '
                     f'Fairness of this metric is between 0 and 1.0 , Please enter again', exc_info=0)
        sys.exit(0)
    else:
        logger.log(99,
                   f'Fairness of this metric is between 0 and {-abs(args.fair_threshold)}')
        ideal_fairness = [0, abs(args.fair_threshold)]
        metric = "Equalised Odds"
        file_path = os.path.join(os.path.dirname(
            os.path.abspath(args.output)), metric + '.png')

    aaod = average_abs_equalised_odd(y_privileged, y_unprivileged, preds_privileged,
                                     preds_unprivileged, args.clf_threshold)
    plot_fairness(aaod, args, metric=metric,
                  ideal_fairness=ideal_fairness, file_path=file_path)
    with open(args.output, 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Fairness Plot', 'Equalised Odds'])
        writer.writerow([file_path, aaod])


def main():
    parser = argparse.ArgumentParser(
        description='Equalised Odds\n'
                    '\n'
                    'Equalised Odds,\n' +
                    'This metric is computed as average of absolute difference between '
                    'false positive rate and true positive rate'
                    'for unprivileged and privileged groups. \n' +
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input',
        help='path to input csv file, output result shown in the evaluation tab (csv) default=output_result.csv',
        required=True, default='output_result.csv')

    parser.add_argument(
        '-t', '--target_variable',
        help='specify the actual target(label) variable (variable)',
        required=True)

    parser.add_argument(
        '-p', '--output_variable',
        help="specify the predicted output variable, expect sigmoid output (variable) default=y'",
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
        help='best optimal classification threshold, default=0.5',
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
        '-o', '--output', help='path to output file (csv) default=equalised_odds.csv',
        required=True, default='equalised_odds.csv')
    parser.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)
    logger.log(99, 'completed successfully.')


if __name__ == '__main__':
    main()
