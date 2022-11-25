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
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nnabla import logger


def extract_col(header, table, col_name):
    '''
    Extract the columns
    '''
    try:
        col_index = header.index(col_name)
    except BaseException:
        cols = [j for j, col in enumerate(header)if col_name == col.split('__')[
            0].split(':')[0]]
        if len(cols) == 0:
            logger.critical(
                f'Variable {col_name} is not found in the dataset.')
        else:
            col_index = cols[0]
            logger.log(
                99, f'Column {header[col_index]} was used instead of column {col_name}.')
    return [float(line[col_index]) for line in table]


def calc_precision_recall(y_label, y_pred):
    '''
    Calculate precission and recall
    Args:
        y_label     (list) : ground truth target values
        y_pred      (list) : estimated targets as returned by a classifier

    return:
        precision (float): precision value
        recall    (float): recall value
    '''
    tp_count = 0
    fp_count = 0
    fn_count = 0
    length = len(y_label)
    for i in range(length):
        if y_label[i] == y_pred[i] == 1:
            tp_count += 1
        if y_pred[i] == 1 and y_label[i] != y_pred[i]:
            fp_count += 1
        if y_pred[i] == 0 and y_label[i] != y_pred[i]:
            fn_count += 1
    try:
        precision = tp_count / (tp_count + fp_count)
    except:
        precision = 1
    try:
        recall = tp_count / (tp_count + fn_count)
    except:
        recall = 1
    return precision, recall


def auc_score(recall, precision):
    '''
    Compute Area Under Curve(AUC)
    Args:
        precision (list): computed precision values
        recall (list): computed recall values
    Return:
        auc_score (float): auc_score
    '''
    height = 0.5*(np.array(precision[1:])+np.array(precision[:-1]))
    width = -((np.array(recall[1:])-np.array(recall[:-1])))
    auc_score = (height*width).sum()
    return auc_score


def cal_scores(thresholds, y_test_probs, y_test):
    '''
    calculate precision scores, recall scores
    Args:
        thresholds   (float): threshold values
        y_test_probs (list): estimated targets as returned by a classifier
        y_test       (list): target values

    Return:
        precision_scores (list): computed precision values
        recall_scores    (list): computed recall values
    '''
    precision_scores = []
    recall_scores = []
    for thresh in thresholds:
        y_test_preds = []
        for prob in y_test_probs:
            if prob > thresh:
                y_test_preds.append(1)
            else:
                y_test_preds.append(0)
        precision, recall = calc_precision_recall(y_test, y_test_preds)
        precision_scores.append(precision)
        recall_scores.append(recall)
    return precision_scores, recall_scores


def func(args):

    with open(args.input, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        header = next(reader)
        table = list(reader)
    variables = [args.target_variable, args.output_variable]
    y_test = []
    y_test_probs = []
    for col_name in variables:
        col = extract_col(header, table, col_name)
        if len(col):
            if col_name == args.target_variable:
                y_test = col
            else:
                y_test_probs = col
    thresholds = np.arange(0, 1.01, float(args.threshold))
    precision_scores, recall_scores = cal_scores(
        thresholds, y_test_probs, y_test)
    auc = auc_score(recall_scores, precision_scores)
    width = args.width
    height = args.height
    plt.rcParams["figure.figsize"] = (width, height)
    positive = sum(y_test)
    baseline = positive / len(y_test)
    plt.plot([0, 1], [baseline, baseline], linestyle='--',
             color='tan', label='Baseline')
    plt.plot(recall_scores, precision_scores,
             color='blue', label="AUC-PR="+str(auc))
    plt.title('Precision_Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='center left')
    plt.savefig('Precision_Recall_curve.png')
    logger.log(99, 'plot saved!')
    plt.show()
    logger.log(99, 'Precision_Recall curve completed successfully.')


def main():
    '''
    Main
    '''
    parser = argparse.ArgumentParser(
        description='Precision_Recall curve\n\n' +
        ' This plugin draws a two-dimensional plot shows tradeoff between precision and recall' +
        'for different threshold.Higher AUC-PR score indicates better model' +
        'AUC lies between 0 to 1, ' +
        'if it is the poor model AUC near to 0, if it is good model then it goes near 1.' +
        'By Decrease the threshold, get more TP. Baseline of [precision-recall curve] ' +
        'is determined by the ratio of positives (P) and negatives (N) as y = P / (P + N).\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='path to input csv file (csv) default=output_result.csv',
        required=True,
        default='output_result.csv')
    parser.add_argument(
        '-t',
        '--target_variable',
        help="Original label from output_result.csv (variable) default=y",
        required=True,
        default="y")
    parser.add_argument(
        '-p',
        '--output_variable',
        help="Predicted label from output_result.csv (variable) default=y'",
        required=True,
        default="y’")
    parser.add_argument(
        '-wi',
        '--width',
        help='graph width (inches) default=10',
        default=10)
    parser.add_argument(
        '-hi',
        '--height',
        help='graph height (inches) default=8',
        default=8)
    parser.add_argument(
        '-thr',
        '--threshold',
        help='threshold value (float) default=0.02',
        default=0.02)
    parser.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
