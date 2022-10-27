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


def roc_curve(y_label, y_pred, thresholds):
    '''
    calculate TPR and FPR
    Args:
        y_label     (list) : ground truth target values
        y_pred      (list) : estimated targets as returned by a classifier
        thresholds  (array): threshold values

    return:
        fpr (list): False Positive Rate
        tpr (list): True Positive Rate
    '''
    fpr = []
    tpr = []
    positive = sum(y_label)
    negative = len(y_label) - positive
    for thresh in thresholds:
        fp_count = 0
        tp_count = 0
        length = len(y_pred)
        for i in range(length):
            if y_pred[i] > thresh:
                if y_label[i] == 1:
                    tp_count = tp_count + 1
                if y_label[i] == 0:
                    fp_count = fp_count + 1
        fpr.append(fp_count/float(negative))
        tpr.append(tp_count/float(positive))
    return fpr, tpr


def auc_score(fpr, tpr):
    '''
    Compute Area Under Curve
    Args:
        fpr (list): computed fpr values
        tpr (list): computed tpr values
    Return:
        auc_score (float): auc_score
    '''
    height = 0.5*(np.array(tpr[1:])+np.array(tpr[:-1]))
    width = -((np.array(fpr[1:])-np.array(fpr[:-1])))
    auc_score = (height*width).sum()
    return auc_score


def func(args):

    with open(args.input, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        header = next(reader)
        table = list(reader)
    variables = [args.target_variable, args.output_variable]
    y_label = []
    y_pred = []
    for col_name in variables:
        col = extract_col(header, table, col_name)
        if len(col):
            if col_name == args.target_variable:
                y_label = col
            else:
                y_pred = col
    fpr, tpr = roc_curve(y_label, y_pred, np.arange(
        0, 1.01, float(args.threshold)))
    auc = auc_score(fpr, tpr)
    width = args.width
    height = args.height
    plt.rcParams["figure.figsize"] = (width, height)
    plt.scatter(fpr, tpr, linestyle='--', color='blue', label="ROC")
    plt.plot([0, 1], [0, 1], color='navy')
    plt.plot(fpr, tpr, color='tan', label="AUC="+str(auc))
    plt.title('ROC_AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.savefig('roc_auc.png')
    logger.log(99, 'plot saved!')
    plt.show()
    logger.log(99, 'ROC_AUC completed successfully.')


def main():
    '''
    Main
    '''
    parser = argparse.ArgumentParser(
        description='ROC_AUC curve\n\n' +
        'A ROC curve is a plot between tpr and fpr at different thresholds.' +
        'AUC is the area under the ROC. ' +
        'Higher AUC indicates better model that ' +
        'predicts 0 classes as 0 and 1 classes as 1.' +
        'AUC lies between 0 to 1,' +
        'if it is the poor model AUC near to 0, ' +
        'if it is good model then it goes near 1. ' +
        'By Decrease the threshold, ' +
        'get more tp(default = 0.02)\n\n',
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
        help="original label from output_result.csv (variable) default=y",
        required=True,
        default="y")
    parser.add_argument(
        '-p',
        '--output_variable',
        help="Predicted label from output_result.csv (variable) default=y'",
        required=True,
        default="y'")
    parser.add_argument(
        '-wi',
        '--width',
        help='graph width(inches) default=10',
        default=10)
    parser.add_argument(
        '-hi',
        '--height',
        help='graph height(inches) default=8',
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
