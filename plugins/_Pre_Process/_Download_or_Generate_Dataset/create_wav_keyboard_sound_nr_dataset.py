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
import shutil
import zipfile
import glob
import random
import csv
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from nnabla.logger import logger
from nnabla.utils.data_source_loader import download


def create_nr_sample_dataset(path, csv_file_name, files, offset=0):
    logger.log(99, f'Creating {csv_file_name}...')
    table = [['x:noisy', 'y:clean']]
    for i, file in enumerate(tqdm(files)):
        rate, wav = wavfile.read(file)
        clean_file_name = os.path.join(
            path, 'wavfiles', 'clean', f'{i + offset}.wav')
        wavfile.write(os.path.join(path, clean_file_name), rate, wav)

        wav = (wav + np.random.randn(*wav.shape) * np.random.uniform()
               * 512).astype(np.int16)  # max sigma = -30dB
        noisy_file_name = os.path.join(
            path, 'wavfiles', 'noisy', f'{i + offset}.wav')
        wavfile.write(os.path.join(path, noisy_file_name), rate, wav)
        table.append([noisy_file_name, clean_file_name])

    csv_file_name = os.path.join(path, csv_file_name)
    with open(csv_file_name, 'w', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerows(table)


def func(args):
    path = args.output_dir
    original_dataset_path = os.path.join(path, 'wav_keyboard_sound_dataset')

    # Create original training set
    logger.log(99, 'Downloading Wav Keyboard Sound dataset...')

    r = download(
        'https://nnabla.org/sample/sample_dataset/keyboard_sound_dataset.zip')
    with zipfile.ZipFile(r) as zip:
        zip.extractall(original_dataset_path)

    sound_files = glob.glob(os.path.join(original_dataset_path, '*', '*.wav'))
    random.shuffle(sound_files)

    try:
        os.makedirs(os.path.join(path, 'wavfiles'))
        os.makedirs(os.path.join(path, 'wavfiles', 'noisy'))
        os.makedirs(os.path.join(path, 'wavfiles', 'clean'))
    except:
        pass
    create_nr_sample_dataset(path, os.path.join(
        path, 'keyboard_sound_nr_train.csv'), sound_files[:-20])
    create_nr_sample_dataset(path, os.path.join(
        path, 'keyboard_sound_nr_test.csv'), sound_files[-20:], len(sound_files) - 20)
    shutil.rmtree(original_dataset_path)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='KeyboardSoundNR\n\n' +
        'Download KeyboardSound dataset from https://support.dl.sony.com/blogs-ja/dataset/keyboard-sound-dataset/.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=Sound\\wav_keyboard_sound_nr',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
