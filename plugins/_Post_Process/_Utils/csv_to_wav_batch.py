# Copyright 2021 Sony Group Corporation.
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
import argparse
import csv
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile

from nnabla import logger


def get_variable_indexes(target_variables, source_variable_names):
    result = []
    for variable in source_variable_names.split(','):
        if variable in target_variables:
            result.append(target_variables.index(variable))
        else:
            logger.critical(
                f'Variable {variable} is not found in the input CSV file.')
            raise
    return result


def func(args):
    # Open input CSV file
    logger.log(99, 'Loading input CSV file ...')
    with open(args.input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        table = [row for row in reader]
    variables = [v.split(':')[0] for v in header]
    input_csv_path = os.path.dirname(args.input_csv)

    # Settings for each variable
    variable_index = get_variable_indexes(variables, args.input_variable)[0]

    # Restore
    logger.log(99, 'Processing CSV files ...')
    header.append(header[variable_index] + '_wav')
    os.chdir(input_csv_path)
    output_path = os.path.dirname(args.output)

    # Input CSV file line loop
    for i, line in enumerate(tqdm(table)):
        wav = (np.loadtxt(line[variable_index]) * 32768).astype(np.int16)
        wav_file_name_csv = os.path.join(
            'wavfiles', f'{i // 1000:08}', f'{i % 1000:03}.wav')
        wav_file_name = os.path.join(output_path, wav_file_name_csv)
        if not os.path.exists(os.path.dirname(wav_file_name)):
            os.makedirs(os.path.dirname(wav_file_name))
        wavfile.write(wav_file_name, args.sampling_rate, wav)
        line.append(wav_file_name)

    logger.log(99, 'Saving CSV file...')
    with open(os.path.join(args.output), 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(table)

    logger.log(99, 'Restore split image completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='CSV to wav (batch)\n\n' +
        'Convert CSV files in the input dataset CSV file to wav file.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input-csv',
        help='dataset CSV file containing CSV files (csv)',
        required=True)
    parser.add_argument(
        '-v',
        '--input-variable',
        help="variables of the CSV to be converted in the dataset CSV (variable) default=y'",
        required=True)
    parser.add_argument(
        '-r',
        '--sampling_rate',
        help='Sampling rate of wav file (int) default=44100',
        required=True,
        type=int)
    parser.add_argument(
        '-o', '--output', help='output csv file (file) default=wav_files.csv', required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
