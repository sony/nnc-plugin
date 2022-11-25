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
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from nnabla import logger


def plot_fairness(model_fairness, args,
                  ideal_fairness=[-0.10, 0.10],
                  metric="Bias Amplification", file_path=None):
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


def get_bias_amplification_attribute_task(running_labels, running_preds):
    """
    compute bias amplification for attribute-task
    References:
        [1] Directional Bias Amplification(https://arxiv.org/abs/2102.12594)
        [2] Fair Attribute Classification through Latent Space
        De-biasing(https://arxiv.org/abs/2012.01469)
    Args:
        running_labels : labels on test set
        running_preds : predictions on test set
    Returns:
         bias amplification
    """
    bog_tilde = np.zeros((2, 2))
    bog_gt_g = np.zeros((2, 2))
    for i, objs in enumerate([running_labels, running_preds]):
        un_privileged = np.where(objs[:, 2] == 1)[0]  # Unprivileged class
        privileged = np.where(objs[:, 1] == 1)[0]  # Privileged class
        un_favourable = np.where(objs[:, 0] == 0)[0]  # Unfavourable class
        favourable = np.where(objs[:, 0] == 1)[0]  # Favourable class
        if i == 0:
            bog_tilde[0][0] = len(set(un_favourable) & set(un_privileged))
            bog_tilde[0][1] = len(set(un_favourable) & set(privileged))
            bog_tilde[1][0] = len(set(favourable) & set(un_privileged))
            bog_tilde[1][1] = len(set(favourable) & set(privileged))
        elif i == 1:
            bog_gt_g[0][0] = len(set(un_favourable) & set(un_privileged))
            bog_gt_g[0][1] = len(set(un_favourable) & set(privileged))
            bog_gt_g[1][0] = len(set(favourable) & set(un_privileged))
            bog_gt_g[1][1] = len(set(favourable) & set(privileged))

    total_images_train = np.sum(bog_tilde)
    data_bog = bog_tilde / np.sum(bog_tilde, axis=0)
    pred_bog = bog_gt_g / np.sum(bog_tilde, axis=0)
    p_t_a = bog_tilde / np.sum(bog_tilde, axis=0)
    p_t = np.sum(bog_tilde, axis=1) / total_images_train
    diff = np.zeros_like(data_bog)
    for i in range(len(data_bog)):
        for j in range(len(data_bog[0])):
            diff[i][j] = pred_bog[i][j] - data_bog[i][j]
            indicator = np.sign(p_t_a[i][j] - p_t[i])
            if indicator == 0:
                diff[i][j] = 0
            elif indicator == -1:
                diff[i][j] = - diff[i][j]
    value = np.nanmean(diff)
    return value


def get_bias_amplification(targets, pred,
                           privilegedvariable, un_privilegedvariable,
                           classification_threshold):
    """
    BA measures how much more often a target attribute is predicted with a
    protected attribute than the ground truth value.
    References:
        [1] Directional Bias Amplification(https://arxiv.org/abs/2102.12594)
        [2] Fair Attribute Classification through Latent Space
        De-biasing(https://arxiv.org/abs/2012.01469)
    Args:
        targets (numpy.ndarray) : actual target label
        pred (numpy.ndarray) : predicted label,
        privilegedvariable (numpy.ndarray) : list of privileged variables
        un_privilegedvariable (numpy.ndarray) : list of un privileged variables
        classification_threshold (float) : classification threshold

    Returns:
        Bias amplification
    """

    test_labels = np.zeros((targets.shape[0], 3))
    test_labels[:, 0] = targets
    test_labels[:, 1] = privilegedvariable
    test_labels[:, 2] = un_privilegedvariable
    test_pred = np.zeros((targets.shape[0], 3))
    pred = np.where(pred > classification_threshold, 1, 0)
    test_pred[:, 0] = pred
    test_pred[:, 1] = privilegedvariable
    test_pred[:, 2] = un_privilegedvariable
    auc_bias = get_bias_amplification_attribute_task(test_labels, test_pred)
    return round(auc_bias, 4)


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
    fair_flag = abs(args.fair_threshold) <= 1  # check the fairness value
    if not fair_flag:
        logger.error(f'Fairness value out of range! '
                     f'Fairness of this metric is between -1.0 and 1.0 , Please enter again', exc_info=0)
        sys.exit(0)
    else:
        logger.log(99,
                   f'Fairness of this metric is between {-abs(args.fair_threshold)} and {-abs(args.fair_threshold)}')
        ideal_fairness = [-abs(args.fair_threshold), abs(args.fair_threshold)]
        metric = "Bias Amplification"
        file_path = os.path.join(os.path.dirname(
            os.path.abspath(args.output)), metric + '.png')

    ba = get_bias_amplification(
        y_actual, clf_out, privileged_variable, unprivileged_variable, args.clf_threshold)
    plot_fairness(ba, args, metric=metric,
                  ideal_fairness=ideal_fairness, file_path=file_path)
    with open(args.output, 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Fairness Plot', 'Bias Amplification'])
        writer.writerow([file_path, ba])


def main():
    parser = argparse.ArgumentParser(
        description='Bias Amplification\n'
                    '\n'
                    'Bias Amplification,\n' +
                    'Bias amplification measures how much more often a target attribute is predicted with '
                    'protected attribute than the ground truth value.\n' +
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
        '-o', '--output', help='path to output file (csv) default=bias_amplification.csv',
        required=True, default='bias_amplification.csv')
    parser.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)
    logger.log(99, 'completed successfully.')


if __name__ == '__main__':
    main()
