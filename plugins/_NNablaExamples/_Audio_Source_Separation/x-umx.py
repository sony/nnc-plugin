# Copyright 2024 Sony Group Corporation.
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
import sys
import os
import argparse
import csv
import shutil
import subprocess
from nnabla import logger
from nnabla.utils.data_source_loader import download, get_data_home


def func(args):
    logger.log(99, 'Checking the required modules...')
    original_path = os.getcwd()
    output_file = os.path.abspath(args.output)
    input_file = os.path.abspath(args.input)

    # Check input file and ffmpeg
    import pydub
    try:
        audio = pydub.AudioSegment.from_file(input_file)
    except:
        if not os.path.exists(input_file):
            logger.critical(f'Input file "{input_file}" not found.')
        else:
            logger.critical('ffmpeg is not installed or ffmpeg/bin path is not added to the PATH of environment variable in the Engine tab in the Setup window of NNC. Please install ffmpeg on your PC, add ffmpeg/bin path to the PATH of environment variable in the Engine tab in the Setup window and try again.')
        exit(1)

    # Download model
    logger.log(99, 'Downloading model...')

    model_url = 'https://nnabla.org/pretrained-models/ai-research-code/x-umx/x-umx.h5'
    download(model_url, open_file=False)
    model_path = os.path.join(get_data_home(), 'x-umx.h5')

    # Process separation
    logger.log(99, 'Processing ...')
    logger.log(
        99, 'This process can take a lot of time if the waveform to be processed is long.')
    output_dir = os.path.join('.', os.path.splitext(
        os.path.basename(args.output))[0])
    try:
        os.makedirs(output_dir)
    except:
        pass
    output_abs_dir = os.path.abspath(output_dir)
    code = os.path.join(os.path.dirname(__file__), '..',
                        'ai-research-code', 'x-umx', 'test.py')
    python_path = os.path.dirname(code)
    sys.path.append(python_path)
    os.chdir(python_path)
    print(output_abs_dir)
    command = ['python', code, '--inputs', input_file,
               '--model', model_path, '--out-dir', output_abs_dir]
    subprocess.call(command)
    os.chdir(original_path)

    # Save all the images
    csv_data = [['x:wav']]
    input_base_name = os.path.splitext(os.path.basename(args.input))[0]
    for track in ['bass', 'drums', 'vocals', 'other']:
        filename = os.path.join(
            '.', os.path.basename(output_dir), f'{track}.wav')
        if os.path.exists(filename):
            csv_data.append([filename])
        else:
            logger.critical('Processing failed.')
            exit(1)

    # Save CSV file
    with open(output_file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)

    logger.log(99, 'Plugin completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='X-UMX Music Source Separation\n\n' +
        'All for One and One for All: Improving Music Separation by Bridging Networks\n' +
        'Ryosuke Sawata, Stefan Uhlich, Shusuke Takahashi, Yuki Mitsufuji\n' +
        'https://arxiv.org/abs/2010.04228\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='input audio file (file)',
        required=True)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output csv file(csv), default=x-umx_output.csv',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
