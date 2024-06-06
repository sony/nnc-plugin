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

    model_cache_dir = os.path.join(get_data_home(), 'centernet_models')
    try:
        os.makedirs(model_cache_dir)
    except:
        pass

    num_layer = 18 if args.architecture == 'resnet' else 34
    model_file_name = f'{args.architecture}_{num_layer}_{args.dataset}_fp.h5'
    model_url = 'https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/ctdet/' + model_file_name
    model_file_name = os.path.join(model_cache_dir, model_file_name)
    download(model_url, output_file=model_file_name, open_file=False)

    # Process image
    logger.log(99, 'Processing ...')
    output_file = os.path.join(os.getcwd(), args.output)
    code = os.path.join(os.path.dirname(__file__), '..', 'nnabla-examples',
                        'object-detection', 'centernet', 'src', 'demo.py')
    command = ['python', code, 'ctdet', '--dataset', args.dataset, '--arch', args.architecture, '--num_layers',
               str(num_layer), '--trained_model_path', model_file_name, '--demo', args.input, '--debug', '1', '--save_dir', os.path.dirname(output_file)]
    subprocess.call(command)

    shutil.move(os.path.join(os.path.dirname(
        output_file), 'ctdet.jpg'), output_file)

    logger.log(99, 'Plugin completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='CenterNet\n\n' +
        'Objects as Points\n' +
        'Xingyi Zhou, Dequan Wang, Philipp Krahenbuhl\n' +
        'https://arxiv.org/abs/1904.07850\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='input image(image)',
        required=True)
    parser.add_argument(
        '-a',
        '--architecture',
        help='backbone architecture(option:resnet,dlav0), default=dlav0',
        default='dlav0')
    parser.add_argument(
        '-d',
        '--dataset',
        help='dataset which the model is trained on(option:coco,pascal), default=pascal',
        default='pascal')
    parser.add_argument(
        '-o',
        '--output',
        help='path to output jpg file(file), default=centernet_output.jpg',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
