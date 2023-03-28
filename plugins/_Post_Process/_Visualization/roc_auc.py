# Copyright 2022,2023 Sony Group Corporation.
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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


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


def func(args):

    with open(args.input, 'r', encoding='utf8') as file:
        reader = csv.reader(file)
        header = next(reader)
        table = list(reader)
    out_var = [c for c in (args.output_variable).split(",")]
    if len(out_var) == 1:
        class_of_interest = [0]
    else:
        class_of_interest = [int(sub.split('__')[1]) for sub in out_var]
    index_list = []
    for i in out_var:
        if i in header:
            index_list.append(header.index(i))
        else:
            print(i, "is not found in the table")
            return
    plt.rcParams["figure.figsize"] = (args.width, args.height)
    y_test = []
    y_test_probs = []
    y_test = extract_col(header, table, args.target_variable)
    target_label = label_binarize(y_test, classes=np.unique(y_test))
    classes = np.unique(y_test)
    class_id = []
    [class_id.append(np.flatnonzero(classes == id)[0])
     for id in class_of_interest]
    [y_test_probs.append(list(row[k] for k in index_list)) for row in table]
    predicted_label = np.array(y_test_probs).astype(np.float64)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for count, i in enumerate(class_id):
        fpr[i], tpr[i], _ = roc_curve(
            target_label[:, i], predicted_label[:, count])
        roc_auc[i] = auc(fpr[i], tpr[i])
        if len(index_list) == 1:
            plt.plot(fpr[i], tpr[i],
                     label='ROC curve (area = %0.2f) ' % (roc_auc[i]))
        else:
            plt.plot(fpr[i], tpr[i],  label='ROC curve (area = %0.2f) for label %s' % (
                roc_auc[i], out_var[count]))
    plt.plot([0, 0], [1, 0], c=".7", label='Perfect Classifier')
    plt.plot([1, 1], c=".7")
    plt.title('Receiver operating characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.savefig('ROC_curve.png')
    plt.show()


def main():
    '''
    Main
    '''
    parser = argparse.ArgumentParser(
        description='ROC curve\n\n' +
        'A ROC curve is a plot between true positive rate and false positive rate at different thresholds.' +
        'AUC lies between 0 to 1,' +
        'if it is the poor model AUC near to 0, ' +
        'if it is good model then it goes near 1. \n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='path to input csv file (csv) default = output_result.csv',
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
        help="Predicted labels separated with commas from output_result.csv (variable) For multiclassification - (y'__0,y'__1, ... , y'__n) default=y'",
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
    parser.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
