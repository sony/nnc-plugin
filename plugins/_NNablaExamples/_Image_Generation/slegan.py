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
import os
import shutil
import argparse
import subprocess
from nnabla import logger
from nnabla.utils.data_source_loader import download, get_data_home


def func(args):
    # Download model
    logger.log(99, 'Downloading models...')

    model_cache_dir = os.path.join(
        get_data_home(), 'slegan_models', args.model)
    try:
        os.makedirs(model_cache_dir)
    except:
        pass

    model_url = 'https://nnabla.org/pretrained-models/nnabla-examples/GANs/slegan/' + args.model + '/'
    download(model_url + 'Gen_iter100000.h5',
             output_file=os.path.join(model_cache_dir, 'Gen_iter100000.h5'), open_file=False)
    download(model_url + 'GenEMA_iter100000.h5', output_file=os.path.join(
        model_cache_dir, 'GenEMA_iter100000.h5'), open_file=False)

    # Generate image
    logger.log(99, 'Generating image...')
    result_dir = os.path.join('.', 'result')
    result_dir_exists = os.path.exists(result_dir)
    generator = os.path.join(os.path.dirname(
        __file__), '..', 'nnabla-examples', 'image-generation', 'slegan', 'generate.py')
    command = ['python', generator, '--model-load-path',
               model_cache_dir, '--batch-size', '1']
    subprocess.call(command)
    shutil.move(os.path.join(result_dir, 'tmp',
                'Image-Tile', '000000.png'), args.output)
    if not result_dir_exists:
        shutil.rmtree(result_dir)

    logger.log(99, 'Plugin completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='SLE-GAN\n\n' +
        'Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis\n' +
        'Bingchen Liu, Yizhe Zhu, Kunpeng Song, Ahmed Elgammal\n' +
        'https://arxiv.org/abs/2101.04775\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m',
        '--model',
        help='model(option:cat,bridge,fountain,obama,panda,temple,wuzhen,dog),default=cat',
        required=True)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output png file(file), default=slegan_output.png',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
