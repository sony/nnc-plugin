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
from collections import OrderedDict
import google.protobuf.text_format as text_format

import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla import logger
from nnabla.ext_utils import get_extension_context
from nnabla.utils.download import download
from nnabla.models.utils import get_model_home
from nnabla.utils.save import save
from nnabla.utils import nnabla_pb2
from nnabla.utils.image_utils import imread
from nnabla.utils.load import base_load
from nnabla.utils.cli.train import train_command

# autopep8: off
sys.path.append(os.path.join(os.path.dirname(__file__),
                '..', '_NNablaExamples', 'nnabla-examples'))
import vision_and_language.clip.clip as clip
# autopep8: on


def func(args):
    # Open config
    logger.log(99, 'Loading config...')
    proto_config = nnabla_pb2.NNablaProtoBuf()
    with open(args.config, mode='r') as f:
        text_format.Merge(f.read(), proto_config)
    nnp = base_load([args.config], prepare_data_iterator=False,
                    exclude_parameter=True)

    # Open dataset
    # training image size, validation image size, runtime image size, crop image size
    input_shape = [None]*4
    batch_size = proto_config.dataset[0].batch_size
    logger.log(99, 'Checking datasets...')
    for i in range(2):  # training and validation datasets
        with open(proto_config.dataset[i].uri, encoding='utf_8_sig') as f:
            reader = csv.reader(f)
            header = next(reader)
            if len(header) != 2:
                logger.critical(
                    f'The dataset must consist of two variables, x and y. Header of the specified data = {header}')
                sys.exit(1)
            num_class = 1
            for row in reader:
                c = int(row[1]) + 1
                if c > num_class:
                    num_class = c
                if input_shape[i] is None:
                    img_sample = imread(os.path.join(os.path.split(
                        proto_config.dataset[i].uri)[0], row[0]), channel_first=True)
                    if len(img_sample.shape) == 2:
                        logger.critical(
                            'This plug-in does not support monochrome images.')
                        sys.exit(1)
                    elif img_sample.shape[0] != 3:
                        logger.critical(
                            'This plug-in supports RGB image only.')
                        sys.exit(1)
                    else:
                        input_shape[i] = img_sample.shape
    input_shape[2] = input_shape[1]
    input_shape[3] = (input_shape[1][0], int(
        input_shape[1][1] * args.crop_ratio), int(input_shape[1][2] * args.crop_ratio))

    # load model
    if args.model.lower() == 'vitb16':
        model_name = 'ViT-B-16'
    elif args.model.lower() == 'vitb32':
        model_name = 'ViT-B-32'
    elif args.model.lower() == 'vitl14':
        model_name = 'ViT-L-14'
    else:
        logger.critical(f'{args.model} is not supported.')
        sys.exit(1)
    path_nnp = os.path.join(
        get_model_home(), 'clip', f'{model_name}.h5')
    url = f'https://nnabla.org/pretrained-models/nnabla-examples/vision-and-language/clip/{model_name}.h5'
    logger.log(99, f'Downloading {model_name} from {url}...')
    dir_nnp = os.path.dirname(path_nnp)
    if not os.path.isdir(dir_nnp):
        os.makedirs(dir_nnp)
    download(url, path_nnp, open_file=False, allow_overwrite=False)

    logger.log(99, f'Loading {model_name}...')
    clip.load(path_nnp)

    logger.log(99, 'Preparing config for transfer learning...')
    # prepare networks
    mean = nn.parameter.get_parameter_or_create(
        name="mean",
        shape=(1, 3, 1, 1),
        initializer=(np.asarray(
            [0.48145466, 0.4578275, 0.40821073]) * 255.0).reshape(1, 3, 1, 1),
        need_grad=False)
    std = nn.parameter.get_parameter_or_create(
        name="std",
        shape=(1, 3, 1, 1),
        initializer=(np.asarray(
            [0.26862954, 0.26130258, 0.27577711]) * 255.0).reshape(1, 3, 1, 1),
        need_grad=False)

    networks = []
    for i, network_name in enumerate(['training', 'validation', 'validation5', 'runtime']):
        network = {'name': network_name, 'batch_size': batch_size}

        x = nn.Variable((batch_size,) + input_shape[0 if i == 0 else 1])
        t = nn.Variable((batch_size, 1))
        normalized_x = (x - mean) / std
        if network_name == 'training':
            min_scale = np.sqrt(
                (224 * 224) / (input_shape[i][1] * input_shape[i][2]))
            h = F.image_augmentation(normalized_x, shape=(3, 224, 224), min_scale=min_scale, max_scale=min_scale *
                                     args.max_zoom_ratio, aspect_ratio=args.max_aspect_ratio, flip_lr=args.lr_flip)
        else:
            h = F.slice(normalized_x, start=(0, (input_shape[1][1] - input_shape[3][1]) // 2, (input_shape[1][2] - input_shape[3][2]) // 2), stop=(
                3, (input_shape[1][1] - input_shape[3][1]) // 2 + input_shape[3][1], (input_shape[1][2] - input_shape[3][2]) // 2 + input_shape[3][2]), step=(1, 1, 1))
            h = F.interpolate(h, output_size=(224, 224))

        h = clip.encode_image(h)
        h = PF.affine(h, num_class, name='last_fc')

        if network_name == 'runtime':
            network['names'] = {'x': x}
            y = F.softmax(h, axis=1)
            network['outputs'] = {'y\'': y}
        else:
            network['names'] = {'x': x, 'y': t}
            if network_name == 'training':
                l = F.softmax_cross_entropy(h, t, axis=1)
                network['outputs'] = {'loss': l}
            elif network_name == 'validation':
                e = F.top_n_error(h, t)
                network['outputs'] = {'error': e}
            elif network_name == 'validation5':
                e = F.top_n_error(h, t, n=5)
                network['outputs'] = {'error': e}

        networks.append(network)

    # prepare parameters for solver
    def parameters_to_be_optimized(param):
        if param == 'mean' or param == 'std':
            return False
        elif 'last_fc' not in param and args.train_param != 'all':
            return False
        else:
            return True
    params = nn.get_parameters()
    params_new = OrderedDict()
    for param in list(params.keys()):
        if parameters_to_be_optimized(param):
            params_new[param] = params[param]
        else:
            params[param].need_grad = False
    solver = nnp.optimizers[list(nnp.optimizers.keys())[0]].solver
    solver.set_parameters(params_new)

    # save model to nnp file
    logger.log(99, f'Saving model...')
    contents = {
        'global_config': {'default_context': get_extension_context('cudnn')},
        'networks': networks,
        'datasets': [
            {'name': 'Training',
             'uri': proto_config.dataset[0].uri,
             'cache_dir': proto_config.dataset[0].cache_dir,
             'variables': networks[0]['names'],
             'overwrite_cache': proto_config.dataset[0].overwrite_cache,
             'create_cache_explicitly': proto_config.dataset[0].create_cache_explicitly,
             'shuffle': True,
             'batch_size': batch_size,
             'no_image_normalization': True},
            {'name': 'Validation',
             'uri': proto_config.dataset[1].uri,
             'cache_dir': proto_config.dataset[1].cache_dir,
             'variables': networks[0]['names'],
             'overwrite_cache': proto_config.dataset[1].overwrite_cache,
             'create_cache_explicitly': proto_config.dataset[1].create_cache_explicitly,
             'shuffle': False,
             'batch_size': batch_size,
             'no_image_normalization': True
             }],
        'training_config':
            {'max_epoch': proto_config.training_config.max_epoch,
             'iter_per_epoch': proto_config.training_config.iter_per_epoch,
             'save_best': proto_config.training_config.save_best},
        'optimizers': [
            {'name': 'optimizer',
             'network': 'training',
             'dataset': 'Training',
             'data_variables': {'x': 'x', 'y': 'y'},
             'solver': solver,
             'weight_decay': proto_config.optimizer[0].solver.weight_decay,
             'lr_decay': proto_config.optimizer[0].solver.lr_decay,
             'lr_decay_interval': proto_config.optimizer[0].solver.lr_decay_interval,
             'update_interval': proto_config.optimizer[0].update_interval}],
        'monitors': [
            {'name': 'train_error',
             'network': 'validation5',
             'dataset': 'Validation',
             'data_variables': {'x': 'x', 'y': 'y'}},
            {'name': 'valid_error',
             'network': 'validation',
             'dataset': 'Validation',
             'data_variables': {'x': 'x', 'y': 'y'}}],
        'executors': [
            {'name': 'runtime',
             'network': 'runtime',
             'no_image_normalization': True,
             'data': ['x'],
             'output': ['y\'']}]}
    args.config = os.path.join(args.output_dir, 'config.nnp')
    save(args.config, contents, variable_batch_size=False)

    # Training
    logger.log(99, f'Executing training...')
    args.outdir = args.output_dir
    args.resume = False
    args.param = None
    args.enable_ooc = False
    args.ooc_gpu_memory_size = None
    args.context = None
    args.ooc_window_length = None

    train_command(args)


def main():
    parser = argparse.ArgumentParser(
        description='Transfer learning (ViT)\n' +
        '\n' +
        'Reuse the pre-trained ViT(Vision Transformer) model to train the image classifier.\n' +
        'Specify both Training and Validation datasets in the DATASET tab,\n' +
        'Specify Epoch, Batch Size and Updater in the CONFIG tab and execute this plug-in.\n' +
        '\n' +
        'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale\n' +
        'Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby\n' +
        'https://arxiv.org/abs/2010.11929', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-c',
        '--config',
        help='config file(nntxt) default=net.nntxt',
        required=True,
        default='net.nntxt')
    parser.add_argument(
        '-o',
        '--output_dir',
        help='output_dir(dir)',
        required=True)
    parser.add_argument(
        '-m',
        '--model',
        help='model(option:vitb16,vitb32,vitl14),default=vitb32',
        default='vitb32', required=True)
    parser.add_argument(
        '-t',
        '--train_param',
        help='Parameters to be trained(option:all,last_fc_only),default=all', default='last_fc_only')
    parser.add_argument(
        '-z',
        '--max_zoom_ratio',
        help='zoom augmentation ratio during training 1.0~(float),default=3.5',
        type=float, default=3.5)
    parser.add_argument(
        '-a',
        '--max_aspect_ratio',
        help='aspect ratio augmentation ratio during training 1.0~(float),default=1.3',
        type=float, default=1.3)
    parser.add_argument(
        '-f',
        '--lr_flip',
        help='LR flip augmentation during training(bool),default=True',
        action='store_true')
    parser.add_argument(
        '-crr',
        '--crop_ratio',
        help='inference crop ratio ~1.0(float),default=0.875',
        type=float, default=0.875)
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
