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
import sys
import numpy as np
from sklearn import metrics
from nnabla import logger
from utils.utils import *


def reject_option_classification(y_true, y_predicted_score,
                                 privileged_group, unprivileged_group, metric_name="demographic_parity",
                                 metric_upper_bound=0.10, metric_lower_bound=0.05):
    """
    Reject Option-Based Classification is a post-processing techinque to 
    mitigate the bias at the model prediction stage, enhancing the favourable 
    outcomes to the unprivileged groups and unfavorable outcomes to privileged 
    groups in a confidence band arround the decision boundary with highest uncertainty.
    
    Args:
        y_true (numpy.ndarray) : ground truth (correct) target values.
        y_predicted_score (numpy.ndarray) : estimated probability predictions (targets scores)
                                            as returned by the classifier.
        privileged_group (numpy.ndarray): list of privileged group values.
        unprivileged_group (numpy.ndarray): list of unprivileged group values.
        metric_name (str) : name of the metric to use for the optimization
                            (demographic_parity, equalised_odd,
                            equal_opportunity).
        metric_upper_bound (float) : upper bound of constraint on the metric value.
        metric_lower_bound (float) : lower bound of constraint on the metric value.

    Returns:
        ROC_margin (float): critical region boundary.
        classification_threshold (float) : optimal classification threshold.
        best_accuracy (float) : accuracy of the model. 
        fairness (float) : fairness of the model.

    """

    low_classification_threshold = 0.01  # smallest classification threshold
    high_classification_threshold = 0.99  # highest classification threshold
    # number of classification threshold b/w low class threshold and high class threshold
    number_classification_threshold = 100
    number_ROC_margin = 50  # number of relevant ROC margins to be used in the optimization search

    fair_metric_array = np.zeros(number_classification_threshold * number_ROC_margin)
    balanced_accuracy_array = np.zeros_like(fair_metric_array)
    ROC_margin_array = np.zeros_like(fair_metric_array)
    classification_threshold_array = np.zeros_like(fair_metric_array)
    cond_privileged_group = privileged_group == 1.0
    cond_unprivileged_group = unprivileged_group == 1.0
    count = 0
    # Iterate through class thresholds
    for class_thresh in np.linspace(low_classification_threshold,
                                    high_classification_threshold,
                                    number_classification_threshold):

        classification_threshold = class_thresh
        if class_thresh <= 0.5:
            low_ROC_margin = 0.0
            high_ROC_margin = class_thresh
        else:
            low_ROC_margin = 0.0
            high_ROC_margin = (1.0 - class_thresh)

        # Iterate through ROC margins
        for ROC_margin in np.linspace(
                low_ROC_margin,
                high_ROC_margin,
                number_ROC_margin):
            ROC_margin = ROC_margin
            # Predict using the current threshold and margin

            y_pred = predict(y_predicted_score, classification_threshold,
                             ROC_margin, cond_privileged_group, cond_unprivileged_group)
            acc = metrics.accuracy_score(y_true, y_pred)
            ROC_margin_array[count] = ROC_margin
            classification_threshold_array[count] = classification_threshold
            balanced_accuracy_array[count] = acc
            y_privileged = y_true[cond_privileged_group]
            y_unprivileged = y_true[cond_unprivileged_group]
            preds_privileged = y_pred[cond_privileged_group]
            preds_unprivileged = y_pred[cond_unprivileged_group]
            if metric_name == "demographic_parity":
                dpd = get_demographic_parity(y_privileged, y_unprivileged,
                                             preds_privileged, preds_unprivileged)
                fair_metric_array[count] = dpd
            elif metric_name == "equalised_odd":
                aaod = get_equalised_odds(y_privileged, y_unprivileged,
                                          preds_privileged, preds_unprivileged)
                fair_metric_array[count] = aaod
            elif metric_name == "equal_opportunity":
                eod = get_equal_opportunity_diff(y_privileged, y_unprivileged,
                                                 preds_privileged, preds_unprivileged)
                fair_metric_array[count] = eod
            else:
                sys.exit(0)
            count += 1

    rel_inds = np.logical_and(fair_metric_array >= metric_lower_bound,
                              fair_metric_array <= metric_upper_bound)
    if any(rel_inds):
        best_ind = np.where(balanced_accuracy_array[rel_inds]
                            == np.max(balanced_accuracy_array[rel_inds]))[0][0]
    else:
        # warn("Unable to satisfy fairness constraints")
        rel_inds = np.ones(len(fair_metric_array), dtype=bool)
        best_ind = np.where(fair_metric_array[rel_inds]
                            == np.min(fair_metric_array[rel_inds]))[0][0]
        logger.critical(f"Unable to satisfy fairness constraints : {metric_lower_bound} to {metric_upper_bound}",
                        exc_info=1)
        sys.exit(0)

    ROC_margin = ROC_margin_array[rel_inds][best_ind]
    classification_threshold = classification_threshold_array[rel_inds][best_ind]
    best_accuracy = balanced_accuracy_array[rel_inds][best_ind]
    fairness = fair_metric_array[rel_inds][best_ind]

    return ROC_margin, classification_threshold, best_accuracy, fairness


def predict(y_predicted_score, classification_threshold, roc_margin, cond_priv, cond_unpriv):
    """
    Obtain fair predictions with ROC method
    Args:
        y_predicted_score (numpy.ndarray): estimated probability predictions
                                           (targets score) as returned by the classifier.
        classification_threshold (float) : optimal classification threshold.
        roc_margin (float) : critical region boundary.
        cond_priv (numpy.ndarray) : privileged group.
        cond_unpriv (numpy.ndarray) : unprivileged group.
    Returns:
        y_pred (numpy.ndarray) : predictions using ROC method.
    """

    y_pred = np.where(y_predicted_score > classification_threshold, 1, 0)

    # Indices of critical region around the classification boundary
    critical_region_indices = np.logical_and(
        y_predicted_score <= classification_threshold + roc_margin,
        y_predicted_score > classification_threshold - roc_margin)

    # Indices of privileged and unprivileged groups
    y_pred[np.logical_and(critical_region_indices,
                          cond_priv)] = 0
    y_pred[np.logical_and(critical_region_indices,
                          cond_unpriv)] = 1

    return y_pred


def func(args):
    logger.log(99, 'Loading input ...')
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        in_header = next(reader)
        in_table = [[col.strip() for col in row]
                    for row in reader if len(row) > 0]

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

    ROC_margin, clf_thr, acc, fairness = reject_option_classification(y_actual, clf_out,
                                                                      privileged_variable, unprivileged_variable,
                                                                      metric_name=args.fair_metric,
                                                                      metric_upper_bound=args.metric_ub,
                                                                      metric_lower_bound=args.metric_lb)

    with open(args.output, 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Classification Threshold', 'ROC Margin', 'Accuracy', args.fair_metric])
        writer.writerow([clf_thr, ROC_margin, acc, fairness])


def restricted_fairness(x):
    """
    check fairness limits 
    """

    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < -1.0 or x > 1.0:
        raise argparse.ArgumentTypeError("Fairness metric value : %r not in range [-1.0, 1.0]" % (x,))
    return x


def main():
    parser = argparse.ArgumentParser(
        description='Reject Option-Based Classification\n'
                    '\n'
                    'Reject Option Based Classification,\n' +
                    'Changes predictions from a classifier to make them fairer. '
                    'Enhances favorable outcomes to unprivileged groups and '
                    'unfavorable outcomes to privileged groups in a confidence band around '
                    'the decision boundary with highest uncertainty.\n\n'
                    'Citation: \nKamiran, Faisal, Asim Karim, and Xiangliang Zhang. '
                    'Decision theory for discrimination-aware classification. In 2012 IEEE 12th '
                    'International Conference on Data Mining, pp. 924-929. IEEE, 2012. '
                    '\n' +

                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input',
        help='path to input csv file (csv) default=output_result.csv',
        required=True, default='output_result.csv')

    parser.add_argument(
        '-t', '--target_variable',
        help='actual target(label) variable (variable)',
        required=True)

    parser.add_argument(
        '-p', '--output_variable',
        help="predicted output variable, expect sigmoid output (variable) default=y'",
        required=True, default="y'")
    parser.add_argument(
        '-zp', '--privileged_variable',
        help='privileged variable from the input CSV file (variable)',
        required=True)
    parser.add_argument(
        '-zu', '--unprivileged_variable',
        help='unprivileged variable from the input CSV file (variable)',
        required=True)

    parser.add_argument(
        '-m',
        '--fair_metric',
        help='metric to use for the optimization(option:demographic_parity,equal_opportunity,'
             'equalised_odd),default=demographic_parity',
        default='demographic_parity', required=True)

    parser.add_argument(
        '-fair_ub',
        '--metric_ub',
        help='upper bound of constraint on the metric value, default value is 0.1, default=0.1',
        type=restricted_fairness, default=0.1)

    parser.add_argument(
        '-fair_lb',
        '--metric_lb',
        help='lower bound of constraint on the metric value, default value is 0.01, default=-0.1',
        type=restricted_fairness, default=-0.1)

    parser.add_argument(
        '-o', '--output', help='path to output file (csv) default=roc.csv',
        required=True, default='roc.csv')
    parser.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)
    logger.log(99, 'completed successfully.')


if __name__ == '__main__':
    main()
