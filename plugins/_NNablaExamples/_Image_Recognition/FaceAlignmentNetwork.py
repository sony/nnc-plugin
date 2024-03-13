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

    model_cache_dir = os.path.join(
        get_data_home(), 'face_alignment_network_models')
    try:
        os.makedirs(model_cache_dir)
    except:
        pass

    model_file_names = ['2DFAN4_NNabla_model.h5'] if args.landmarks_type == '2d' else [
        '3DFAN4_NNabla_model.h5', 'Resnet_Depth_NNabla_model.h5']
    model_file_urls = ['https://nnabla.org/pretrained-models/nnabla-examples/face-alignment/' +
                       model_file_name for model_file_name in model_file_names]
    model_file_names = [os.path.join(
        model_cache_dir, model_file_name) for model_file_name in model_file_names]
    for model_file_name, model_file_url in zip(model_file_names, model_file_urls):
        logger.log(99, os.path.basename(model_file_name))
        download(model_file_url, output_file=model_file_name, open_file=False)

    # Process image
    logger.log(99, 'Processing ...')
    output_file = os.path.join(os.getcwd(), args.output)
    code = os.path.join(os.path.dirname(__file__), '..', 'nnabla-examples',
                        'facial-keypoint-detection', 'face-alignment', 'model_inference.py')
    if args.landmarks_type == '2d':
        command = ['python', code, '--model', model_file_names[0],
                   '--test-image', args.input, '--output', output_file]
    else:
        command = ['python', code, '--landmarks-type-3D', '--model', model_file_names[0],
                   '--resnet-depth-model', model_file_names[1], '--test-image', args.input, '--output', output_file]
    subprocess.call(command)

    if os.path.exists(output_file):
        logger.log(99, 'Plugin completed successfully.')
    else:
        logger.critical('Processing failed.')
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Face Alignment Network\n\n' +
        'How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)\n' +
        'Adrian Bulat, Georgios Tzimiropoulos\n' +
        'https://arxiv.org/abs/1703.07332\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='input image(image)',
        required=True)
    parser.add_argument(
        '-l',
        '--landmarks_type',
        help='landmarks_type(option:2d,3d), default=2d',
        default='2d')
    parser.add_argument(
        '-o',
        '--output',
        help='path to output image file(file), default=face_alignment_network_output.png',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
