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


import os
import csv
import sys
import subprocess
from nnabla.logger import logger
from nnabla.utils.data_source_loader import download


def main():
    path = os.path.abspath(os.path.dirname(__file__))

    # Download
    file_name = os.path.join(path, f'german.data')
    data = download(
        f'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data').read()
    with open(file_name, 'wb') as f:
        f.write(data)

    # Create original dataset file
    csv_file_name = 'german_credit_original.csv'
    logger.log(99, 'Creating "{}"... '.format(csv_file_name))

    with open(os.path.join(path, csv_file_name), 'w', newline='') as fw:
        writer = csv.writer(fw)
        writer.writerow(['Status of existing checking account', 'Duration in month', 'Credit history', 'Purpose', 'Credit amount', 'Savings account/bonds', 'Present employment since', 'Installment rate in percentage of disposable income', 'Personal status and sex',
                        'Other debtors / guarantors', 'Present residence since', 'Property', 'Age in years', 'Other installment plans', 'Housing', 'Number of existing credits at this bank', 'Job', 'Number of people being liable to provide maintenance for', 'Telephone', 'foreign worker', 'Good / bad'])
        with open(os.path.join(path, file_name)) as fr:
            for line in fr:
                row = line.split()
                row[-1] = 'Good' if row[-1] == '1' else 'bad'
                writer.writerow(row)

    # Create text classification dataset
    logger.log(99, 'Creating CSV files... ')
    command = [sys.executable, '../../../../../libs/plugins/_Pre_Process/_Create_Dataset/simple_tabular_data.py', '-i', './{}'.format(
        csv_file_name), '-b', 'Good / bad', '-t', '-o', './', '-g', 'log.txt', '-f1', 'german_credit_train.csv', '-r1', '80', '-f2', 'german_credit_test.csv', '-r2', '20']
    p = subprocess.call(command)
    os.remove(file_name)

    logger.log(99, 'Dataset creation completed successfully.')


if __name__ == '__main__':
    main()
