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
import argparse
import math
import csv
import tqdm
import numpy as np

from nnabla.logger import logger


def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def create_csv(target_dir, dataset_name, num_data, data_len, with_abnormal_data):
    # CSV header

    csv_data = [['x:sin_wave']]
    if with_abnormal_data:
        csv_data[0].append('y:abnormal')

    # Create output dir
    makedirs(os.path.join(target_dir, dataset_name))

    # Create samples
    for data_index in tqdm.tqdm(range(num_data)):
        wav = []
        phase = np.random.rand()
        freq = np.random.rand() + 1
        amplitude = np.random.rand() * 0.5 + 0.5
        abnormal = with_abnormal_data and data_index % 2
        abnormal_phase = np.random.randint(
            data_len - 3) if abnormal else data_len
        abnormal_amplitude = np.random.rand() * 0.1 + 0.1
        noise_amplitude = np.random.rand() * 0.01
        for i in range(data_len):
            value = math.sin((i * 1.0 / data_len * freq + phase) * math.pi * 2) * amplitude + \
                np.random.randn() * noise_amplitude
            if i >= abnormal_phase and i < abnormal_phase + 3:
                value += np.random.randn() * abnormal_amplitude
            wav.append([value])
        # Save data CSV
        data_file_name = os.path.join(
            target_dir, dataset_name, str(data_index) + '.csv')
        with open(data_file_name, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(wav)
        if with_abnormal_data:
            csv_data.append([data_file_name, str(abnormal)])
        else:
            csv_data.append([data_file_name])

    # Save CSV
    with open(os.path.join(target_dir, dataset_name + '.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)


def func(args):
    path = args.output_dir
    np.random.seed(0)

    # Create training dataset (normal data only)
    logger.log(99, 'Converting training dataset (unsupervised) ... ')
    create_csv(path, "training_unsupervised", 3000, 128, False)

    # Create training dataset (normal and abnormal data)
    logger.log(99, 'Converting training dataset (supervised) ... ')
    create_csv(path, "training_labeled", 3000, 128, True)

    # Create validation dataset (normal and abnormal data)
    logger.log(99, 'Converting validation dataset ... ')
    create_csv(path, "validation_labeled", 100, 128, True)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='SyhtheticSinWave_AnomalyDetection\n\n' +
        'Create "Syhthetic sin wave anomaly detection" dataset.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=synthetic_data\\sin_wave_anomaly_detection',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
