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
import shutil
import argparse
import math

import zipfile
import google.protobuf.text_format as text_format
import numpy as np

import nnabla as nn
import nnabla.functions as F
from nnabla import logger
import nnabla_ext.cpu
from nnabla.ext_utils import get_extension_context
from nnabla.utils import nnabla_pb2
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from nnabla.utils.progress import configure_progress, progress
from nnabla.utils.save import save


def get_model(model, batch_size, feature_ratio):
    try:
        nn.set_default_context(get_extension_context('cudnn'))
        x = nn.Variable()
        F.relu(x)
    except:
        logger.warn('Fallback to CPU context.')
        nn.set_default_context(nnabla_ext.cpu.context())

    layer_shapes = [28, 14]
    layers = [None] * len(layer_shapes)
    if model == 'resnet18':
        from nnabla.models.imagenet import ResNet18
        model = ResNet18()
    elif model == 'resnext50':
        from nnabla.models.imagenet import ResNeXt50
        model = ResNeXt50()
    elif model == 'senet':
        from nnabla.models.imagenet import SENet
        model = SENet()
    else:
        logger.critical(f'{model} is not supported.')
        sys.exit(1)
    sys.stdout.flush()

    x = nn.Variable((batch_size,) + model.input_shape)
    y = model(x, training=False)

    class enum_layers:
        def __init__(self):
            self.functions = []

        def __call__(self, f):
            self.functions.append(f)
    callback = enum_layers()
    y.visit(callback)
    for f in callback.functions:
        for i, size in enumerate(layer_shapes):
            if f.outputs[0].shape[-2:] == (size, size):
                layers[i] = f
    assert (None not in layers)

    hs = [layer.outputs[0] for layer in layers]
    if feature_ratio < 1.0:
        hs = [h[:, :int(h.shape[1] * feature_ratio), :, :] for h in hs]
    hs = [F.average_pooling(h, (3, 3), stride=(1, 1), pad=(1, 1)) for h in hs]
    for i, layer in enumerate(layers):
        logger.log(
            99, f'Use {hs[i].shape[1:]} of the output of {layer.name} of shape {layer.outputs[0].shape[1:]} as layer {i + 1}')
    h = F.concatenate(hs[0], F.unpooling(hs[1], (2, 2)), axis=1)
    return x, h


def func(args):
    configure_progress(os.path.join(args.output_dir, 'progress.txt'))
    # Open config
    logger.log(99, 'Loading config...')
    proto = nnabla_pb2.NNablaProtoBuf()
    with open(args.config, mode='r') as f:
        text_format.Merge(f.read(), proto)
    batch_size = proto.dataset[0].batch_size

    # Prepare model
    logger.log(99, f'Preparing {args.model} model...')
    x, h = get_model(args.model, batch_size, args.feature_ratio)

    # Prepare training dataset
    logger.log(99, 'Preparing training dataset...')
    data_iterator = (lambda: data_iterator_csv_dataset(
        uri=proto.dataset[0].uri,
        batch_size=batch_size,
        shuffle=False,
        normalize=False,
        with_memory_cache=False,
        with_file_cache=False))

    # Forward
    logger.log(99, 'Calculating features...')
    with data_iterator() as di:
        data_size = di.size
        feature = np.ndarray((data_size,) + h.shape[1:], dtype=np.float32)
        try:
            vind = di.variables.index('x')
        except:
            logger.critical(
                f'Variable "x" is not found in the training dataset.')
            raise
        index = 0
        while index < data_size:
            progress(f'{index}/{data_size}', index * 1.0 / data_size)
            data = di.next()[vind]
            try:
                x.d = data.reshape(x.shape)
            except:
                logger.critical(
                    f'The image size supported by {args.model} is {x.shape[1:]}, but the size of the image in the training dataset is {data.shape[1:]}.')
                raise

            try:
                h.forward()
            except RuntimeError:
                logger.critical(
                    f'If you run out of memory, reduce the value of batch size.')
                raise

            data_size_1 = batch_size if index + batch_size < data_size else data_size - index
            feature[index:index + data_size_1] = h.d[:data_size_1]
            index += batch_size
    feature = feature.transpose(0, 2, 3, 1).reshape(
        feature.shape[0] * feature.shape[2] * feature.shape[3], feature.shape[1])
    progress(None)

    # Create anomary detection model
    memory_size = int(len(feature) * args.sample_ratio)
    logger.log(
        99, f'Creating anomary detection model (memory size = {memory_size})...')

    memory = np.ndarray((memory_size,) + feature.shape[1:], dtype=np.float32)
    for i in range(memory_size):
        progress(f'{i + 1}/{memory_size}', (i ** 2) / (memory_size ** 2))
        max_index = 0
        max_distance = 0.0
        for i2 in range(i):
            distance = np.sqrt(
                ((memory[i2, np.newaxis] - feature) ** 2).sum(axis=1)).min(axis=0)
            if distance > max_distance:
                max_distance = distance
                max_index = i
        memory[i] = feature[max_index]
        feature[max_index] = feature[len(feature) - 1]
        feature = feature[:len(feature) - 1]
    progress(None)

    memory = memory.T.reshape((1, memory.shape[1], memory.shape[0], 1, 1))
    memory_param = nn.parameter.get_parameter_or_create(
        name="memory",
        shape=memory.shape,
        need_grad=False,
        initializer=memory)
    y = F.min(F.pow_scalar(F.sum((h.reshape(
        (h.shape[0], h.shape[1], 1, h.shape[2], h.shape[3])) - memory_param) ** 2, axis=1), 0.5), axis=1)

    # Create anomary detection model
    logger.log(99, 'Saving anomary detection model...')
    contents = {
        'global_config': {'default_context': nn.get_current_context()},
        'networks': [
            {'name': 'network',
             'batch_size': 1,
             'outputs': {'x\'': y},
             'names': {'x': x}}],
        'executors': [
            {'name': 'runtime',
             'network': 'network',
             'data': ['x'],
             'output': ['x\'']}]}
    result_config_file = os.path.join(args.output_dir, 'results.nntxt')
    save(result_config_file, contents)
    result_model_file = os.path.join(args.output_dir, 'results.nnp')
    save(result_model_file, contents)
    result_proto = nnabla_pb2.NNablaProtoBuf()
    with open(result_config_file, mode='r') as f:
        text_format.Merge(f.read(), result_proto)
    result_proto.executor[0].no_image_normalization = True
    with open(result_config_file, mode='w') as f:
        text_format.PrintMessage(result_proto, f)

    result_model_file_tmp = result_model_file + '.tmp'
    with zipfile.ZipFile(result_model_file, 'r') as zr:
        with zipfile.ZipFile(result_model_file_tmp, 'w') as zw:
            for info in zr.infolist():
                ext = os.path.splitext(info.filename)[1].lower()
                if ext == ".nntxt" or ext == ".prototxt":
                    zw.write(result_config_file, arcname=info.filename)
                else:
                    bin = zr.read(info.filename)
                    zw.writestr(info, bin)
    shutil.move(result_model_file_tmp, result_model_file)

    logger.log(99, 'Training Completed.')


def main():
    parser = argparse.ArgumentParser(
        description='PatchCore\n' +
        '\n' +
        'Towards Total Recall in Industrial Anomaly Detection\n' +
        'Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Scholkopf, Thomas Brox, Peter Gehler\n' +
        'https://arxiv.org/abs/2106.08265', formatter_class=argparse.RawTextHelpFormatter)
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
        default='ResNet18')
    parser.add_argument(
        '-fr',
        '--feature_ratio',
        help='ratio of features used for anomaly detection(float),default=0.25',
        default=0.25,
        type=float)
    parser.add_argument(
        '-sr',
        '--sample_ratio',
        help='ratio of samples used for anomaly detection(float),default=0.01',
        default=0.01,
        type=float)
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
