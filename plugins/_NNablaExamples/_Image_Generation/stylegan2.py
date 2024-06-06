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
import sys
import argparse
import csv
from nnabla import logger
from nnabla.utils.data_source_loader import download, get_data_home

# autopep8: off
sys.path.append(os.path.join(os.path.dirname(__file__), '..',
                             'nnabla-examples', 'image-generation', 'stylegan2'))
from generate import *


# autopep8: on


def func(args):

    # Download model
    logger.log(99, 'Downloading model...')

    model_url = 'https://nnabla.org/pretrained-models/nnabla-examples/GANs/stylegan2/styleGAN2_G_params.h5'
    download(model_url, open_file=False)

    # Prepare model
    logger.log(99, 'Preparing model...')
    ctx = get_extension_context("cudnn")
    nn.set_default_context(ctx)
    num_layers = 18
    nn.load_parameters(os.path.join(get_data_home(), 'styleGAN2_G_params.h5'))

    # Generate images
    logger.log(99, 'Generating images...')
    rnd = np.random.RandomState(args.latent_seed)
    z = rnd.randn(args.batch_size, 512)

    nn.set_auto_forward(True)

    style_noise = nn.NdArray.from_numpy_array(z)
    style_noises = [style_noise for _ in range(2)]

    rgb_output = generate(args.batch_size, style_noises, args.noise_seed,
                          mix_after=7, truncation_psi=args.truncation_psi)

    images = convert_images_to_uint8(rgb_output, drange=[-1, 1])

    # Save all the images
    csv_data = [['x:image']]
    output_dir = os.path.join('.', os.path.splitext(
        os.path.basename(args.output))[0] + '_images')
    try:
        os.makedirs(output_dir)
    except:
        pass
    for i in range(args.batch_size):
        filename = os.path.join(output_dir, f'seed{i:08}.png')
        csv_data.append([filename])
        imsave(filename, images[i], channel_first=True)

    # Save CSV file
    with open(args.output, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)

    logger.log(99, 'Plugin completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='StyleGAN2\n\n' +
        'Analyzing and Improving the Image Quality of StyleGAN\n' +
        'Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila\n' +
        'https://arxiv.org/abs/1912.04958\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-l',
        '--latent_seed',
        help='the seed for noise input z. this drastically changes the result (int),default=600',
        default=600,
        type=int)
    parser.add_argument(
        '-t',
        '--truncation_psi',
        help='the value for truncation trick. (float),default=0.5',
        default=0.5,
        type=float)
    parser.add_argument(
        '-n',
        '--noise_seed',
        help='the seed for stochasticity input. this slightly changes the result. (int),default=500',
        default=500,
        type=int)
    parser.add_argument(
        '-b',
        '--batch_size',
        help='number of images to generate (int),default=8',
        default=8,
        type=int)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output csv file(csv), default=stylegan2_output.csv',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
