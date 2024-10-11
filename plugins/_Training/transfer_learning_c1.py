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

    if args.model == 'resnet50':
        model_name = 'resnet50'
    elif args.model == 'se_resnext101':
        model_name = 'se_resnext101'
    else:
        logger.critical(f'{args.model} is not supported.')
        sys.exit(1)

    path_nnp = os.path.join(
        get_model_home(), 'c1_models', f'{model_name}_c1.nnp')
    url = f'https://nnabla.org/pretrained-models/nnp_models/c1_models/{model_name}_c1.nnp'
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
    # Copy runtime network to validation network
    for i, n in enumerate(proto_pretrained.network):
        # runtime network
        if n.name == proto_pretrained.executor[0].network_name:
            monitor1 = proto_pretrained.monitor.add()
            monitor1.name = 'train_error'
            monitor1.network_name = 'top5error'
            monitor1.dataset_name.append('Validation')
            monitor2 = proto_pretrained.monitor.add()
            monitor2.name = 'valid_error'
            monitor2.network_name = 'top1error'
            monitor2.dataset_name.append('Validation')
            proto_pretrained.network.append(n)
            proto_pretrained.network[-1].name = proto_pretrained.monitor[0].network_name
            proto_pretrained.network.append(n)
            proto_pretrained.network[-1].name = proto_pretrained.monitor[1].network_name
            break

    # Edit network
    training_parameters = []
    for i, n in enumerate(proto_pretrained.network):
        n.batch_size = batch_size

        # x -> mulscalar -> addscalar -> (slice) -> image_augmentation -> transpose-> pad
        mul_scalar = n.function.add()
        mul_scalar.input.append('x')
        mul_scalar.name = 'MulScalar'
        mul_scalar.type = 'MulScalar'
        mul_scalar.mul_scalar_param.val = 0.01735
        mul_scalar.output.append('MulScalar')
        mul_scalar_variable = n.variable.add()
        mul_scalar_variable.name = mul_scalar.name
        mul_scalar_variable.type = 'Buffer'
        mul_scalar_variable.shape.dim.append(-1)
        if n.name == 'training':
            mul_scalar_variable.shape.dim.extend(input_shape[0])
        else:
            mul_scalar_variable.shape.dim.extend(input_shape[1])

        add_scalar = n.function.add()
        add_scalar.input.append('MulScalar')
        add_scalar.name = 'AddScalar'
        add_scalar.type = 'AddScalar'
        add_scalar.add_scalar_param.val = -1.99
        add_scalar.output.append('AddScalar')
        add_scalar_variable = n.variable.add()
        add_scalar_variable.name = add_scalar.name
        add_scalar_variable.type = 'Buffer'
        add_scalar_variable.shape.CopyFrom(mul_scalar_variable.shape)

        if n.name != 'training' and args.crop_ratio < 1.0:
            slice = n.function.add()
            slice.input.append('AddScalar')
            slice.name = 'Slice'
            slice.type = 'Slice'
            slice.slice_param.start.extend(
                [0, (input_shape[1][1] - input_shape[2][1]) // 2, (input_shape[1][2] - input_shape[2][2]) // 2])
            slice.slice_param.stop.extend(
                [3, slice.slice_param.start[1] + input_shape[2][1], slice.slice_param.start[2] + input_shape[2][2]])
            slice.slice_param.step.extend([1, 1, 1])
            slice.output.append('Slice')
            slice_variable = n.variable.add()
            slice_variable.name = slice.name
            slice_variable.type = 'Buffer'
            slice_variable.shape.dim.append(-1)
            slice_variable.shape.dim.extend(input_shape[2])

        image_augmentation = n.function.add()
        if n.name == 'training':
            image_augmentation.input.append('AddScalar')
            image_augmentation.image_augmentation_param.flip_lr = args.lr_flip
            image_augmentation.image_augmentation_param.aspect_ratio = args.max_aspect_ratio
            image_augmentation.image_augmentation_param.min_scale = np.sqrt(
                (224 * 224) / (input_shape[0][1] * input_shape[0][2]))
            image_augmentation.image_augmentation_param.max_scale = image_augmentation.image_augmentation_param.min_scale * args.max_zoom_ratio
        elif args.crop_ratio >= 1.0:  # no center crop
            image_augmentation.input.append('AddScalar')
            image_augmentation.image_augmentation_param.aspect_ratio = 1.0
            image_augmentation.image_augmentation_param.min_scale = np.sqrt(
                (224 * 224) / (input_shape[0][1] * input_shape[0][2]))
            image_augmentation.image_augmentation_param.max_scale = image_augmentation.image_augmentation_param.min_scale
        else:
            image_augmentation.input.append('Slice')  # center crop
            image_augmentation.image_augmentation_param.aspect_ratio = 1.0
            image_augmentation.image_augmentation_param.min_scale = np.sqrt(
                (224 * 224) / (input_shape[2][1] * input_shape[2][2]))
            image_augmentation.image_augmentation_param.max_scale = image_augmentation.image_augmentation_param.min_scale
        image_augmentation.name = 'ImageAugmentation'
        image_augmentation.type = 'ImageAugmentation'
        image_augmentation.image_augmentation_param.shape.dim.extend([
                                                                     3, 224, 224])
        image_augmentation.image_augmentation_param.pad.dim.extend([0, 0])
        image_augmentation.image_augmentation_param.angle = 0
        image_augmentation.image_augmentation_param.distortion = 0
        image_augmentation.image_augmentation_param.flip_ud = False
        image_augmentation.image_augmentation_param.brightness = 0
        image_augmentation.image_augmentation_param.brightness_each = False
        image_augmentation.image_augmentation_param.contrast = 1
        image_augmentation.image_augmentation_param.contrast_center = 0
        image_augmentation.image_augmentation_param.contrast_each = False
        image_augmentation.image_augmentation_param.noise = 0
        image_augmentation.image_augmentation_param.seed = -1
        image_augmentation.output.append('ImageAugmentation')
        image_augmentation_variable = n.variable.add()
        image_augmentation_variable.name = image_augmentation.name
        image_augmentation_variable.type = 'Buffer'
        image_augmentation_variable.shape.dim.extend([-1, 3, 224, 224])

        transpose = n.function.add()
        transpose.input.append('ImageAugmentation')
        transpose.name = 'Transpose'
        transpose.type = 'Transpose'
        transpose.transpose_param.axes.extend([0, 2, 3, 1])
        transpose.output.append('Transpose')
        transpose_variable = n.variable.add()
        transpose_variable.name = transpose.name
        transpose_variable.type = 'Buffer'
        transpose_variable.shape.dim.extend([-1, 224, 224, 3])

        pad = n.function.add()
        pad.input.append('Transpose')
        pad.name = 'Pad'
        pad.type = 'Pad'
        pad.pad_param.pad_width.extend([0, 1])
        pad.pad_param.mode = 'constant'
        pad.pad_param.constant_value = 0
        pad.output.append('Pad')
        pad_variable = n.variable.add()
        pad_variable.name = pad.name
        pad_variable.type = 'Buffer'
        pad_variable.shape.dim.extend([-1, 224, 224, 4])

        used_variable = set()
        for f in n.function:
            if f.name == '@training/Convolution' or f.name == '@runtime/Convolution':
                f.input[0] = 'Pad'
            elif f.type == "Convolution" and f.input[1] == 'fc/fc/conv/W':
                f.name = 'Affine'
                f.type = 'Affine'
                f.input[1] = 'Affine/affine/W'
                f.input[2] = 'Affine/affine/b'
                f.output[0] = f.name
                f.affine_param.base_axis = 1

                affine_variable = n.variable.add()
                affine_variable.name = f.name
                affine_variable.type = 'Buffer'
                affine_variable.shape.dim.extend([-1, num_class])
                w_variable = n.variable.add()
                w_variable.name = f.input[1]
                w_variable.type = 'Parameter'
                w_variable.shape.dim.extend([2048, num_class])
                w_variable.initializer.type = 'Normal'
                w_variable.initializer.multiplier = 0.01
                b_variable = n.variable.add()
                b_variable.name = f.input[2]
                b_variable.type = 'Parameter'
                b_variable.shape.dim.append(num_class)
                b_variable.initializer.type = 'Constant'
                b_variable.initializer.multiplier = 0
                f.ClearField('convolution_param')
            elif n.name == "training" and f.name == '@training/Reshape':
                f.name = 'SoftmaxCrossEntropy'
                f.type = 'SoftmaxCrossEntropy'
                f.input[0] = 'Affine'
                f.input.append('SoftmaxCrossEntropy_T')
                f.output[0] = f.name
                f.softmax_cross_entropy_param.axis = 1
                f.ClearField('reshape_param')

                sce_variable = n.variable.add()
                sce_variable.name = f.output[0]
                sce_variable.type = 'Buffer'
                sce_variable.shape.dim.extend([-1, 1])

                T_variable = n.variable.add()
                T_variable.name = f.input[1]
                T_variable.type = 'Buffer'
                T_variable.shape.dim.extend([-1, 1])
            elif n.name == proto_pretrained.executor[0].network_name and f.name == '@runtime/Reshape':
                f.name = 'y\''
                f.type = 'Softmax'
                f.input[0] = 'Affine'
                f.output[0] = f.name
                f.softmax_param.axis = 1
                f.ClearField('reshape_param')
            elif (n.name == proto_pretrained.monitor[0].network_name or n.name == proto_pretrained.monitor[1].network_name) and f.name == '@runtime/Reshape':
                f.name = 'TopNError'
                f.type = 'TopNError'
                f.input[0] = 'Affine'
                f.input.append('TopNError_T')
                f.output[0] = f.name
                f.top_n_error_param.axis = 1
                f.top_n_error_param.n = 5 if n.name == proto_pretrained.monitor[
                    0].network_name else 1
                f.ClearField('reshape_param')

                tne_variable = n.variable.add()
                tne_variable.name = f.output[0]
                tne_variable.type = 'Buffer'
                tne_variable.shape.dim.extend([-1, 1])

                T_variable = n.variable.add()
                T_variable.name = f.input[1]
                T_variable.type = 'Buffer'
                T_variable.shape.dim.extend([-1, 1])

            used_variable |= set(f.input) | set(f.output)

        for v in n.variable:
            if v.name == 'x':
                v.shape.dim[1:] = input_shape[0]
            elif v.name == 'y':
                v.name = 'y\''
                v.shape.dim[1] = num_class

        for j, v in reversed(list(enumerate(n.variable))):
            if v.name not in used_variable:
                del proto_pretrained.network[i].variable[j]
        for v in n.variable:
            if n.name == 'training' and v.type == 'Parameter':
                training_parameters.append(v.name)

    # Overwrite dataset and config
    if len(proto_config.dataset) < 2 or proto_config.dataset[0].name != 'Training' or proto_config.dataset[1].name != 'Validation':
        logger.critical(
            'This plugin requires two datasets, "Training" and "Validation"')
        sys.exit(1)
    proto_pretrained.dataset.extend(proto_config.dataset)
    for d in proto_pretrained.dataset:
        d.no_image_normalization = True
    proto_pretrained.global_config.CopyFrom(proto_config.global_config)
    proto_pretrained.training_config.CopyFrom(proto_config.training_config)

    # Setup optimizer
    optimizer = proto_pretrained.optimizer.add()
    optimizer.name = 'Optimizer'
    optimizer.update_interval = proto_config.optimizer[0].update_interval
    optimizer.network_name = 'training'
    optimizer.dataset_name.append('Training')
    optimizer.solver.CopyFrom(proto_config.optimizer[0].solver)
    input_variable = optimizer.data_variable.add()
    input_variable.variable_name = 'x'
    input_variable.data_name = 'x'
    label_variable = optimizer.data_variable.add()
    label_variable.variable_name = 'SoftmaxCrossEntropy_T'
    label_variable.data_name = 'y'
    loss_variable = optimizer.loss_variable.add()
    loss_variable.variable_name = 'SoftmaxCrossEntropy'
    for p in training_parameters:
        parameter_variable = optimizer.parameter_variable.add()
        parameter_variable.variable_name = p
        lr_multiplier = 0
        if args.train_param == 'all':
            param_name = p.split('/')[-1]
            if param_name != 'mean' and param_name != 'var':
                lr_multiplier = 1
        elif p == 'Affine/affine/W' or p == 'Affine/affine/b':
            lr_multiplier = 1
        parameter_variable.learning_rate_multiplier = lr_multiplier

    # Setup monitor
    for monitor in proto_pretrained.monitor:
        input_variable = monitor.data_variable.add()
        input_variable.variable_name = 'x'
        input_variable.data_name = 'x'
        label_variable = monitor.data_variable.add()
        label_variable.variable_name = 'TopNError_T'
        label_variable.data_name = 'y'
        monitor_variable = monitor.monitor_variable.add()
        monitor_variable.type = 'Error'
        monitor_variable.variable_name = 'TopNError'

    # Setup executor
    executor = proto_pretrained.executor[0]
    executor.no_image_normalization = True
    del executor.parameter_variable[:]
    for p in training_parameters:
        parameter_variable = executor.parameter_variable.add()
        parameter_variable.variable_name = p
    executor.output_variable[0].variable_name = "y\'"
    executor.output_variable[0].data_name = "y\'"

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
    logger.log(99, f'{args}')

    train_command(args)


def main():
    parser = argparse.ArgumentParser(
        description='Transfer learning (C1)\n' +
        '\n' +
        'Reuse the pre-trained model by C1 dataset (commercially available dataset version 1) to train the image classifier.\n' +
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
        help='model(option:resnet50,se_resnext101),default=resnet50',
        default='resnet50', required=True)
    parser.add_argument(
        '-t',
        '--train_param',
        help='parameters to be trained(option:all,last_fc_only),default=all', default='all')
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
