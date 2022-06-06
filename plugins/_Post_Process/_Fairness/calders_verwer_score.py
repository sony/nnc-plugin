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
import numpy as np
from sklearn import metrics
from nnabla import logger


def calders_verwer_score(output_f, output_m, classification_threshold):
    """
    Calders and Verwer defined a discrimination score,
    by subtracting the conditional probability of the positive class given a sensitive value
    from that given a non-sensitive value.
    Args:
        output_f (numpy.ndarray) : output of unprivileged class
        output_m (numpy.ndarray) : output of privileged class
        classification_threshold (float) : classification threshold
    Returns:
        CV Score (float)
    """
    yf_pred = (output_f >= classification_threshold)
    ym_pred = (output_m >= classification_threshold)
    corr_f = np.sum(yf_pred == True)
    corr_m = np.sum(ym_pred == True)
    p_y1_s1 = corr_f / output_f.shape[0]
    p_y1_s0 = corr_m / output_m.shape[0]
    cv_score = np.abs(p_y1_s0 - p_y1_s1)
    logger.log(99, 'Calder-Verwer discrimination score: %.4f' %
               (cv_score.item()))
    return round(cv_score.item(), 4)


def accuracy_score(actual, predicted):
    """
    Classification accuracy is a metric that summarizes the performance of a
    classification model

    Args:
        actual (numpy.ndarray) : actual output of the model
        predicted (numpy.ndarray) : predicted by the model
    Returns:
        accuracy score (float)
    """
    clf_accuracy = metrics.accuracy_score(actual, predicted) * 100
    return clf_accuracy


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

    def enum_cols(name, col):
        if col not in in_header:
            logger.critical(
                'The column named {} specified by {} is not found in the input dataset'.format(
                    col, name))
        else:
            logger.log(99, '{} : {}'.format(col, name))
        return col

    label_col = enum_cols('label_variable', args.label_variable)
    output_col = enum_cols('output-variable', args.output_variable)
    privileged_col = enum_cols('privileged_variable', args.privileged_variable)
    unprivileged_col = enum_cols(
        'unprivileged_variable', args.unprivileged_variable)

    def extract_col(col):
        col_index = in_header.index(col)
        return [float(in_table[i][col_index]) for i in range(len(in_table))]

    y_actual = np.array(extract_col(label_col))
    clf_out = np.asarray(extract_col(output_col)).flatten()
    privileged_variable = np.array(extract_col(privileged_col))
    unprivileged_variable = np.array(extract_col(unprivileged_col))
    out_privileged = clf_out[privileged_variable == 1.0]
    out_unprivileged = clf_out[unprivileged_variable == 1.0]
    classification_threshold = 0.5  # default optimal classification threshold

    preds = np.where(clf_out > classification_threshold, 1, 0)
    logger.log(99, type(preds))
    cv_score = calders_verwer_score(
        out_privileged, out_unprivileged, classification_threshold)
    clf_accuracy = accuracy_score(y_actual, preds)
    with open(args.output, 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Object Variable', 'Output variable', 'Privileged variable', 'Unprivileged variable',
                         'Number of samples', 'CV score', 'Accuracy'])
        writer.writerow([args.label_variable, args.output_variable,
                        args.privileged_variable, args.unprivileged_variable,
                        len(in_table), cv_score, clf_accuracy])


def main():
    parser = argparse.ArgumentParser(
        description='CV Score\n'
                    '\n'
                    '"Compute the CV score of the model.\n' +
                    'Calders and Verwer defined a discrimination score,\n' +
                    'by subtracting the conditional probability of the positive class given a sensitive value\n' +
                    'from that given a non-sensitive value\n' +
                    '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input',
        help='path to input csv file, output result shown in the evaluation tab (csv) default=output_result.csv',
        required=True, default='output_result.csv')
    parser.add_argument(
        '-l', '--label_variable', help='specify the label variable (variable) default=y__0:label',
        required=True, default='y__0:label')
    parser.add_argument(
        '-p', '--output_variable',
        help='specify the output variable, in this classification o/p (variable) default=Sigmoid',
        required=True, default='Sigmoid')
    parser.add_argument(
        '-zp', '--privileged_variable',
        help='specify the privileged variable which to compute the '
             'discrimination score (variable) default=z__0:sex=Female',
        required=True, default='z__0:sex=Female')
    parser.add_argument(
        '-zu', '--unprivileged_variable',
        help='specify the unprivileged variable which to compute the '
             'discrimination score (variable) default=z__1:sex=Male',
        required=True, default='z__1:sex=Male')

    parser.add_argument(
        '-n', '--num_samples',
        help='number of samples N, '
             'if num_samples = all, it will take all the samples from the input csv file (variable), default=all',
        required=True, default='all')
    parser.add_argument(
        '-o', '--output', help='path to output file (csv) default=fairness.csv', required=True, default='fairness.csv')
    parser.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)
    logger.log(99, 'completed successfully.')


if __name__ == '__main__':
    main()
