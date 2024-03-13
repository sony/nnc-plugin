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
import nnabla as nn
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
    rnd1 = np.random.RandomState(args.latent_seed)
    z1 = nn.NdArray.from_numpy_array(rnd1.randn(args.batch_size_a, 512))

    rnd2 = np.random.RandomState(args.latent_seed2)
    z2 = nn.NdArray.from_numpy_array(rnd2.randn(args.batch_size_b, 512))

    nn.set_auto_forward(True)

    mix_image_stacks = []
    for i in range(args.batch_size_a):
        image_column = []
        for j in range(args.batch_size_b):
            style_noises = [F.reshape(z1[i], (1, 512)),
                            F.reshape(z2[j], (1, 512))]
            rgb_output = generate(
                1, style_noises, args.noise_seed, args.mix_after, args.truncation_psi)
            image_column.append(convert_images_to_uint8(
                rgb_output, drange=[-1, 1])[0])
        image_column = np.concatenate(
            [image for image in image_column], axis=2)
        mix_image_stacks.append(image_column)
    mix_image_stacks = np.concatenate(
        [image for image in mix_image_stacks], axis=1)

    style_noises = [z1, z1]
    rgb_output = generate(args.batch_size_a, style_noises,
                          args.noise_seed, args.mix_after, args.truncation_psi)
    image_A = convert_images_to_uint8(rgb_output, drange=[-1, 1])
    image_A = np.concatenate([image for image in image_A], axis=1)

    style_noises = [z2, z2]
    rgb_output = generate(args.batch_size_b, style_noises,
                          args.noise_seed, args.mix_after, args.truncation_psi)
    image_B = convert_images_to_uint8(rgb_output, drange=[-1, 1])
    image_B = np.concatenate([image for image in image_B], axis=2)

    top_image = 255 * np.ones(rgb_output[0].shape).astype(np.uint8)

    top_image = np.concatenate((top_image, image_B), axis=2)
    grid_image = np.concatenate((image_A, mix_image_stacks), axis=2)
    grid_image = np.concatenate((top_image, grid_image), axis=1)

    imsave(args.output, grid_image, channel_first=True)

    logger.log(99, 'Plugin completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='StyleGAN2 (Style Mixing)\n\n' +
        'Analyzing and Improving the Image Quality of StyleGAN\n' +
        'Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila\n' +
        'https://arxiv.org/abs/1912.04958\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-l',
        '--latent_seed',
        help='seed for the primary noise input z. This will represent coarse style (int),default=600',
        default=600,
        type=int)
    parser.add_argument(
        '-l2',
        '--latent_seed2',
        help='seed for the secondary noise input z2. This will represent fine style (int),default=500',
        default=500,
        type=int)
    parser.add_argument(
        '-m',
        '--mix_after',
        help='which layer to use the secondary latent code w2. 0~17 (int),default=7',
        default=7,
        type=int)
    parser.add_argument(
        '-n',
        '--noise_seed',
        help='the seed for stochasticity input. this slightly changes the result. (int),default=500',
        default=500,
        type=int)
    parser.add_argument(
        '-t',
        '--truncation_psi',
        help='the value for truncation trick. (float),default=0.5',
        default=0.5,
        type=float)
    parser.add_argument(
        '-ba',
        '--batch_size_a',
        help='number of images made solely from coarse style noise (int),default=1',
        default=1,
        type=int)
    parser.add_argument(
        '-bb',
        '--batch_size_b',
        help='number of images made solely from fine style noise (int),default=4',
        default=4,
        type=int)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output image file(image), default=stylegan2_output.png',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
