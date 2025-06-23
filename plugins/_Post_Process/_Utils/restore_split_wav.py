# Copyright 2021,2022,2023,2024,2025 Sony Group Corporation.
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
    if source_variable_names:
        for variable in source_variable_names.split(','):
            if variable in target_variables:
                result.append(target_variables.index(variable))
            else:
                logger.critical(
                    f'Variable {variable} is not found in the input CSV file.')
                raise
    return result


def load_wav(file_name, default_sampling_freq):
    if os.path.splitext(file_name)[1].lower() == ".csv":
        return default_sampling_freq, (np.loadtxt(file_name) * 32768).astype(np.int16)
    else:
        return wavfile.read(file_name)


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
    wav_index_index = get_variable_indexes(
        variables, args.wav_index_variable)[0]
    variable_index = get_variable_indexes(variables, args.input_variable)[0]
    length_variable_index = get_variable_indexes(
        variables, args.length_variable)[0]
    window_size_variable_index = get_variable_indexes(
        variables, args.window_size_variable)[0]
    overlap_size_variable_index = get_variable_indexes(
        variables, args.overlap_size_variable)[0]
    pos_variable_index = get_variable_indexes(variables, args.pos_variable)[0]
    inherit_col_indexes = get_variable_indexes(variables, args.inherit_cols)

    # Restore
    logger.log(99, 'Processing wav files ...')
    result_header = []
    for col_index in inherit_col_indexes:
        result_header.append(header[col_index])
    result_header.append(header[variable_index])
    result_table = []
    table.sort(key=lambda x: x[1])
    current_path = os.getcwd()
    output_path = os.path.dirname(args.output)

    # Input CSV file line loop
    last_index = -1
    tmp_lines = []

    def restore_wav(lines):
        os.chdir(input_csv_path if input_csv_path else current_path)
        sampling_freq, wav_data = load_wav(
            lines[0][variable_index], args.default_sampling_freq)
        ch = 1 if len(wav_data.shape) == 1 else wav_data.shape[1]

        length = int(lines[0][length_variable_index])
        window_size = int(lines[0][window_size_variable_index])
        overlap_size = int(lines[0][overlap_size_variable_index])
        out_wav_data = np.zeros((length, ch), dtype=wav_data.dtype)
        result_line = []
        for col_index in inherit_col_indexes:
            result_line.append(lines[0][col_index])

        for line in lines:
            sampling_freq, wav_data = load_wav(
                line[variable_index], args.default_sampling_freq)
            wav_data = wav_data.reshape((wav_data.shape[0], ch))
            pos = int(line[pos_variable_index])
            if args.crossfade:
                for i in range(wav_data.shape[0]):
                    p = pos + i
                    if p >= 0 and p < length:
                        if i < overlap_size:
                            out_wav_data[p] += (wav_data[i] * i * 1.0 /
                                                overlap_size).astype(out_wav_data.dtype)
                        elif i >= window_size - overlap_size:
                            out_wav_data[p] += (wav_data[i] * (window_size - i)
                                                * 1.0 / overlap_size).astype(out_wav_data.dtype)
                        else:
                            out_wav_data[p] += wav_data[i]
            else:
                pos += overlap_size
                size = window_size - overlap_size * 2
                if pos + size > length:
                    size = length - pos
                out_wav_data[pos:pos + size,
                             ::] = wav_data[overlap_size:overlap_size + size, ::]
        out_wav_csv_file_name = os.path.join(
            'restore_split_wav',
            f'{last_index//1000:04}', f'{last_index % 1000:03}.wav')
        out_wav_file_name = os.path.join(output_path, out_wav_csv_file_name)
        os.chdir(current_path)
        if not os.path.exists(os.path.dirname(out_wav_file_name)):
            os.makedirs(os.path.dirname(out_wav_file_name))
        wavfile.write(out_wav_file_name, sampling_freq, out_wav_data)
        result_line.append(out_wav_csv_file_name)
        result_table.append(result_line)

    for line in tqdm(table):
        index = int(line[wav_index_index])
        if index != last_index:
            if len(tmp_lines) > 0:
                restore_wav(tmp_lines)
            tmp_lines = []
            last_index = index
        tmp_lines.append(line)

    if len(tmp_lines) > 0:
        restore_wav(tmp_lines)

    logger.log(99, 'Saving CSV file...')
    with open(os.path.join(args.output), 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(result_header)
        writer.writerows(result_table)

    logger.log(99, 'Restore split wav completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Restore Split Wav\n\n' +
        'Restore large wav file from multiple wav files.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input-csv',
        help='dataset CSV file containing split images (csv) default=output_result.csv',
        required=True)
    parser.add_argument(
        '-n',
        '--wav_index-variable',
        help="wav index variable in the dataset CSV (variable) default=index",
        required=True)
    parser.add_argument(
        '-v',
        '--input-variable',
        help="variables of the image to be processed in the dataset CSV (variable) default=y'",
        required=True)
    parser.add_argument(
        '-e',
        '--length-variable',
        help='wav length variable (variable) default=y_original_length',
        required=True)
    parser.add_argument(
        '-w',
        '--window-size-variable',
        help='window size variable (variable) default=y_window_size',
        required=True)
    parser.add_argument(
        '-l',
        '--overlap-size-variable',
        help='overlap size variable (variable) default=y_overlap_size',
        required=True)
    parser.add_argument(
        '-p',
        '--pos-variable',
        help='position variable (variable) default=y_pos',
        required=True)
    parser.add_argument(
        '-c',
        '--crossfade',
        help='crossfade the overlap (bool) default=True',
        action='store_true')
    parser.add_argument(
        '-s',
        '--default-sampling-freq',
        help='default sampling frequency when loading CSV file (int) default=44100',
        type=int)
    parser.add_argument(
        '-o', '--output', help='output csv file (file) default=restored_wav_files.csv', required=True)
    parser.add_argument(
        '-t', '--inherit-cols', help='variables to inherit from input CSV to output CSV (variables) default=# x')
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
