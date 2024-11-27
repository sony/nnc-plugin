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
import zipfile
import google.protobuf.text_format as text_format
import numpy as np

from nnabla import logger
from nnabla.utils import nnabla_pb2
from nnabla.utils.image_utils import imread
from nnabla.utils.save import save
from nnabla.utils.download import download
from nnabla.models.utils import get_model_home, get_model_url_base
from nnabla.utils.cli.train import train_command


def func(args):
    # Open config
    logger.log(99, 'Loading config...')
    proto_config = nnabla_pb2.NNablaProtoBuf()
    with open(args.config, mode='r') as f:
        text_format.Merge(f.read(), proto_config)

    # Open dataset
    # training image size, validation image size, validation crop image size
    input_shape = [None]*3
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
    input_shape[2] = (input_shape[1][0], int(
        input_shape[1][1] * args.crop_ratio), int(input_shape[1][2] * args.crop_ratio))

    # Download model
    if args.model == 'resnet18':
        model_name = 'Resnet-18'
    elif args.model == 'resnext50':
        model_name = 'ResNeXt-50'
    elif args.model == 'senet':
        model_name = 'SENet-154'
    else:
        logger.critical(f'{args.model} is not supported.')
        sys.exit(1)

    path_nnp = os.path.join(
        get_model_home(), 'imagenet', f'{model_name}.nnp')
    url = f'https://nnabla.org/pretrained-models/nnp_models/imagenet/{model_name}/{model_name}.nnp'
    logger.log(99, f'Downloading {model_name} from {url}...')
    dir_nnp = os.path.dirname(path_nnp)
    if not os.path.isdir(dir_nnp):
        os.makedirs(dir_nnp)
    download(url, path_nnp, open_file=False, allow_overwrite=False)

    # Open pre-trained model
    logger.log(99, 'Loading pre-trained model...')
    proto_pretrained = nnabla_pb2.NNablaProtoBuf()
    with zipfile.ZipFile(path_nnp, 'r') as z:
        for info in z.infolist():
            ext = os.path.splitext(info.filename)[1].lower()
            if ext == '.nntxt' or ext == '.prototxt':
                text_format.Merge(z.read(info.filename), proto_pretrained)
            elif ext == '.protobuf' or ext == '.h5':
                z.extract(info.filename, args.output_dir)
                parameter_file = os.path.join(args.output_dir, info.filename)

    # Modify networks
    logger.log(99, 'Preparing config for transfer learning...')
    # Delete runtime network
    for i, n in enumerate(proto_pretrained.network):
        # runtime network
        if n.name == proto_pretrained.executor[0].network_name:
            del proto_pretrained.network[i]
            break
    # Copy validation network to runtime network
    for n in proto_pretrained.network:
        # validation network
        if n.name == proto_pretrained.monitor[1].network_name:
            proto_pretrained.network.append(n)
            proto_pretrained.network[-1].name = proto_pretrained.executor[0].network_name
            break
    # Edit network
    for i, n in enumerate(proto_pretrained.network):
        n.batch_size = batch_size
        used_variable = set()
        for f in n.function:
            if f.type == 'ImageAugmentation':
                if f.image_augmentation_param.flip_lr:  # for training
                    f.image_augmentation_param.min_scale = np.sqrt(
                        (224 * 224) / (input_shape[0][1] * input_shape[0][2]))
                    f.image_augmentation_param.max_scale = f.image_augmentation_param.min_scale * \
                        args.max_zoom_ratio
                    f.image_augmentation_param.flip_lr = args.lr_flip
                    f.image_augmentation_param.aspect_ratio = args.max_aspect_ratio
                else:  # for validation
                    f.image_augmentation_param.min_scale = np.sqrt(
                        (224 * 224) / (input_shape[2][1] * input_shape[2][2]))
                    f.image_augmentation_param.max_scale = f.image_augmentation_param.min_scale
            elif f.type == 'Slice':
                f.slice_param.start[1] = (
                    input_shape[1][1] - input_shape[2][1]) // 2
                f.slice_param.start[2] = (
                    input_shape[1][2] - input_shape[2][2]) // 2
                f.slice_param.stop[1] = f.slice_param.start[1] + \
                    input_shape[2][1]
                f.slice_param.stop[2] = f.slice_param.start[2] + \
                    input_shape[2][2]
            elif f.type == 'Affine':
                f.name = 'Affine2'
                f.input[1] = 'Affine2/affine/W'
                f.input[2] = 'Affine2/affine/b'
                f.output[0] = f.name
            elif n.name == proto_pretrained.executor[0].network_name and f.type == 'TopNError':
                f.name = 'y\''
                f.type = 'Softmax'
                f.input[0] = 'Affine2'
                del f.input[1]
                f.output[0] = f.name
                f.softmax_param.axis = 1
            elif f.type == 'TopNError' or f.type == 'SoftmaxCrossEntropy':
                f.input[0] = 'Affine2'
            used_variable |= set(f.input) | set(f.output)
        for v in n.variable:
            if v.shape.dim[1:] == [3, 480, 480]:
                v.shape.dim[1:] = input_shape[0]
            elif v.shape.dim[1:] == [3, 320, 320]:
                v.shape.dim[1:] = input_shape[1]
            elif v.shape.dim[1:] == [3, 280, 280]:
                v.shape.dim[1:] = input_shape[2]
            elif v.name == 'Affine/affine/W':
                v.name = 'Affine2/affine/W'
                v.shape.dim[1] = num_class
            elif v.name == 'Affine/affine/b':
                v.name = 'Affine2/affine/b'
                v.shape.dim[0] = num_class
            elif v.name == 'Affine':
                v.name = 'Affine2'
                v.shape.dim[1] = num_class
            elif n.name == proto_pretrained.executor[0].network_name and (v.name == 'TopNError' or v.name == 'Top1Error'):
                v.name = 'y\''
        for j, v in reversed(list(enumerate(n.variable))):
            if v.name not in used_variable:
                del proto_pretrained.network[i].variable[j]
    # Overwrite config
    for d, dc in zip(proto_pretrained.dataset, proto_config.dataset):
        if d.name != dc.name:
            logger.critical(
                'This plugin requires two datasets, "Training" and "Validation"')
            sys.exit(1)
        d.batch_size = batch_size
        d.uri = dc.uri
        d.cache_dir = dc.cache_dir
        d.create_cache_explicitly = dc.create_cache_explicitly
    proto_pretrained.global_config.CopyFrom(proto_config.global_config)
    proto_pretrained.training_config.CopyFrom(proto_config.training_config)
    proto_pretrained.optimizer[0].solver.CopyFrom(
        proto_config.optimizer[0].solver)
    proto_pretrained.optimizer[0].update_interval = proto_config.optimizer[0].update_interval
    proto_pretrained.executor[0].num_evaluations = 1
    proto_pretrained.executor[0].no_image_normalization = True
    for pv in proto_pretrained.optimizer[0].parameter_variable:
        if pv.variable_name == 'Affine/affine/W':
            pv.variable_name = 'Affine2/affine/W'
        elif pv.variable_name == 'Affine/affine/b':
            pv.variable_name = 'Affine2/affine/b'
        if 'bn/mean' in pv.variable_name or 'bn/var' in pv.variable_name or (args.train_param != 'all' and 'Affine2' not in pv.variable_name):
            pv.learning_rate_multiplier = 0
        else:
            pv.learning_rate_multiplier = 1
    for pv in proto_pretrained.executor[0].parameter_variable:
        if pv.variable_name == 'Affine/affine/W':
            pv.variable_name = 'Affine2/affine/W'
        elif pv.variable_name == 'Affine/affine/b':
            pv.variable_name = 'Affine2/affine/b'

    # Training
    args.config = os.path.join(args.output_dir, 'results.nntxt')
    with open(args.config, mode='w') as f:
        text_format.PrintMessage(proto_pretrained, f)
    args.outdir = args.output_dir
    args.resume = False
    args.param = parameter_file
    args.enable_ooc = False
    args.ooc_gpu_memory_size = None
    args.context = None
    args.ooc_window_length = None

    train_command(args)


def main():
    parser = argparse.ArgumentParser(
        description='Transfer learning (imagenet)\n' +
        '\n' +
        'Reuse the pre-trained model by ImageNet1k to train the image classifier.\n' +
        'Specify both Training and Validation datasets in the DATASET tab,\n' +
        'Specify Epoch, Batch Size and Updater in the CONFIG tab and execute this plug-in.', formatter_class=argparse.RawTextHelpFormatter)
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
        help='model(option:resnet18,resnext50,senet),default=resnet18',
        default='resnet18', required=True)
    parser.add_argument(
        '-t',
        '--train_param',
        help='Parameters to be trained(option:all,last_fc_only),default=all', default='all')
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
