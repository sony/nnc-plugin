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
import random
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
    logger.log(99, 'Loading original dataset ...')
    with open(args.input_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        table = [row for row in reader]
    variables = [v.split(':')[0] for v in header]
    input_csv_path = os.path.dirname(args.input_csv)

    # Settings for each variable
    variable_indexes = [[], []]
    variable_indexes[0] = get_variable_indexes(variables, args.input_variable1)
    variable_indexes[1] = get_variable_indexes(variables, args.input_variable2)
    window_sizes = [args.window_size1, args.window_size2]
    overlap_sizes = [args.overlap_size1, args.overlap_size2]

    # Add new cols to the header
    extra_header = []
    header.append('index')
    for i in range(2):
        for vi in variable_indexes[i]:
            extra_header.append(header[vi])
            header.extend([f'{variables[vi]}_original_length', f'{variables[vi]}_window_size',
                           f'{variables[vi]}_overlap_size', f'{variables[vi]}_pos'])
    header.extend(extra_header)

    # Comment out original cols
    for i, h in enumerate(header):
        if (i in variable_indexes[0] or i in variable_indexes[1]):
            header[i] = '# ' + header[i]

    # Add original data index to the table
    for i in range(len(table)):
        table[i].append(i)

    # Convert
    logger.log(99, 'Processing wav files ...')
    result_table = [[], []]
    os.chdir(input_csv_path)
    output_path = args.output_dir
    if args.shuffle:
        random.shuffle(table)

    # Input CSV file line loop
    for org_data_index, line in enumerate(tqdm(table)):
        result_lines = []
        extra_lines = []
        wav_output_path = os.path.join(
            output_path, f'{org_data_index//1000:08}')
        if not os.path.exists(wav_output_path):
            os.mkdir(wav_output_path)
        for i in range(2):  # Variable 1 and 2
            for vi in variable_indexes[i]:
                base_size = window_sizes[i] - overlap_sizes[i] * 2
                sampling_freq, wav_data = wavfile.read(line[vi])
                window_num = (wav_data.shape[0] - 1) // base_size + 1
                window_index = 0
                # Window loop
                for window_index in range(window_num):
                    pos = org_pos = - \
                        overlap_sizes[i] + window_index * base_size
                    if pos >= 0:
                        size = window_sizes[i]
                        offset = 0
                    else:
                        offset = -pos
                        size = window_sizes[i] - offset
                        pos = 0
                    if offset + size > wav_data.shape[0]:
                        size = wav_data.shape[0] = offset
                    # Extract window
                    ch = 1 if len(wav_data.shape) == 1 else wav_data.shape[1]
                    wav_data = wav_data.reshape((wav_data.shape[0], ch))
                    out_wav_data = np.zeros(
                        (window_sizes[i], ch), dtype=wav_data.dtype)
                    crop_wav_data = wav_data[pos:pos + size, ::]
                    out_wav_data[offset:offset +
                                 crop_wav_data.shape[0], ::] = crop_wav_data

                    # Add line
                    if len(result_lines) <= window_index:
                        result_lines.append([])
                        extra_lines.append([])
                    out_wav_csv_file_name = os.path.join(
                        f'{org_data_index//1000:08}', f'{variables[vi]}_{org_data_index % 1000:03}_{window_index:08}.wav')
                    out_wav_file_name = os.path.join(
                        output_path, out_wav_csv_file_name)
                    wavfile.write(out_wav_file_name,
                                  sampling_freq, out_wav_data)
                    result_lines[window_index].extend([out_wav_csv_file_name])
                    extra_lines[window_index].extend(
                        [wav_data.shape[0], window_sizes[i], overlap_sizes[i], org_pos])
                    window_index += 1
        result_table[0 if org_data_index < (len(table) * args.ratio1) // 100 else 1].extend(
            [line + e + r for r, e in zip(result_lines, extra_lines)])

    logger.log(99, 'Saving output file 1 ...')
    with open(os.path.join(args.output_dir, args.output_file1), 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        if args.shuffle:
            random.shuffle(result_table[0])
        writer.writerows(result_table[0])

    if args.output_file2 is not None and len(result_table[1]):
        logger.log(99, 'Saving output file 2 ...')
        with open(os.path.join(args.output_dir, args.output_file2), 'w', newline="\n", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(result_table[1])

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Split Wav\n\n' +
        'Split a long wav file into multiple short wav files of a size suitable for signal processing by deep learning.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input-csv',
        help='dataset CSV file containing wav files (csv)',
        required=True)
    parser.add_argument(
        '-v1',
        '--input-variable1',
        help='variables of the wav to be processed in the dataset CSV (variable) default=x',
        required=True)
    parser.add_argument(
        '-w1',
        '--window-size1',
        help='specify window size in pixels (int) default=4096',
        required=True,
        type=int)
    parser.add_argument(
        '-l1',
        '--overlap-size1',
        help='specify the size of the window to be duplicated in samples (int) default=512',
        required=True,
        type=int)
    parser.add_argument(
        '-v2',
        '--input-variable2',
        help='variables of the image to be processed in the dataset CSV (variable) default=y')
    parser.add_argument(
        '-w2',
        '--window-size2',
        help='specify window size in samples (int) default=4096',
        type=int)
    parser.add_argument(
        '-l2',
        '--overlap-size2',
        help='specify the size of the window to be duplicated in samples (int) default=512',
        type=int)
    parser.add_argument(
        '-o', '--output-dir', help='output dir (dir)', required=True)
    parser.add_argument(
        '-s',
        '--shuffle',
        help='shuffle (bool) default=True',
        action='store_true')
    parser.add_argument(
        '-f1',
        '--output_file1',
        help='output file name 1 (csv) default=train.csv',
        required=True,
        default='train.csv')
    parser.add_argument(
        '-r1',
        '--ratio1',
        help='output file ratio 1 (int) default=100',
        type=int,
        required=True)
    parser.add_argument(
        '-f2',
        '--output_file2',
        help='output file name 2 (csv) default=test.csv',
        default='test.csv')
    parser.add_argument(
        '-r2',
        '--ratio2',
        help='output file ratio 2 (int) default=0',
        type=int,
        default=0)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
