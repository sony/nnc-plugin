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
from fileinput import filename
import os
import argparse
import shutil
import subprocess
from nnabla import logger
from nnabla.utils.data_source_loader import download, get_data_home


def func(args):
    # Download model
    logger.log(99, 'Downloading model...')

    model_url = 'https://nnabla.org/pretrained-models/nnabla-examples/esrgan/esrgan_latest_g.h5'
    download(model_url, open_file=False)

    # Process image
    logger.log(99, 'Processing ...')
    code = os.path.join(os.path.dirname(__file__), '..', 'nnabla-examples',
                        'image-superresolution', 'esrgan', 'inference.py')
    command = ['python', code, '--loadmodel',
               os.path.join(get_data_home(), 'esrgan_latest_g.h5'), '--input_image', args.input]
    subprocess.call(command)

    shutil.move(os.path.join('.', 'result.png'), args.output)

    logger.log(99, 'Plugin completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='ESRGAN\n\n' +
        'ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks\n' +
        'Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang\n' +
        'https://arxiv.org/abs/1809.00219\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='input image (image)',
        required=True)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output png file(image), default=esrgan_output.png',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
