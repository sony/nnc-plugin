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

import zipfile
import google.protobuf.text_format as text_format
import numpy as np

import nnabla as nn
from nnabla.utils import nnp_graph
import nnabla.functions as F
from nnabla import logger
import nnabla_ext.cpu
from nnabla.ext_utils import get_extension_context
from nnabla.models.utils import get_model_home, get_model_url_base
from nnabla.utils import nnabla_pb2
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from nnabla.utils.download import download
from nnabla.utils.progress import configure_progress, progress
from nnabla.utils.save import save


def get_model(model, batch_size, feature_ratio):
    nn.set_default_context(get_extension_context('cudnn'))
    try:
        x = nn.Variable()
        F.relu(x)
    except:
        logger.warn('Fallback to CPU context.')
        nn.set_default_context(nnabla_ext.cpu.context())

    layer_shapes = [56, 28, 14]
    layers = [None] * 3
    if model == 'resnet50':
        model_name = 'resnet50'
    elif model == 'se_resnext101':
        model_name = 'se_resnext101'
    else:
        logger.critical(f'{args.model} is not supported.')
        sys.exit(1)
    sys.stdout.flush()

    path_nnp = os.path.join(
        get_model_home(), 'c1_models', f'{model_name}_c1.nnp')
    url = f'https://nnabla.org/pretrained-models/nnp_models/c1_models/{model_name}_c1.nnp'
    logger.log(99, f'Downloading {model_name} from {url}...')
    dir_nnp = os.path.dirname(path_nnp)
    if not os.path.isdir(dir_nnp):
        os.makedirs(dir_nnp)
    download(url, path_nnp, open_file=False, allow_overwrite=False)

    logger.log(99, f'Loading {model_name}...')
    nnp = nnp_graph.NnpLoader(path_nnp)
    graph = nnp.get_network('runtime', batch_size=batch_size)
    x = list(graph.inputs.values())[0]
    y = list(graph.outputs.values())[0]

    x0 = nn.Variable((batch_size, 3, 224, 224))
    h = (x0 * 0.01735) - 1.99
    h = F.pad(F.transpose(h, (0, 2, 3, 1)), (0, 1))
    x.rewire_on(h)

    class enum_layers:
        def __init__(self):
            self.functions = []

        def __call__(self, f):
            self.functions.append(f)
    callback = enum_layers()
    y.visit(callback)
    for f in callback.functions:
        for i, size in enumerate(layer_shapes):
            logger.log(99, f'{f.name}')
            if f.outputs[0].shape[1:3] == (size, size) and f.name[:4] == 'ReLU':
                layers[i] = f
    assert (None not in layers)

    hs = [layer.outputs[0] for layer in layers]
    if feature_ratio < 1.0:
        hs = [F.transpose(
            h[:, :, :, :int(h.shape[-1] * feature_ratio)], (0, 3, 1, 2)) for h in hs]
    for i, layer in enumerate(layers):
        logger.log(
            99, f'Use {hs[i].shape[1:]} of the output of {layer.name} of shape {layer.outputs[0].shape[1:]} as layer {i + 1}')
    return x0, hs


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
    x, hs = get_model(args.model, batch_size, args.feature_ratio)

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
        features = [np.ndarray((data_size,) + h.shape[1:],
                               dtype=np.float32) for h in hs]
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
                nn.forward_all(hs)
            except RuntimeError:
                logger.critical(
                    f'If you run out of memory, reduce the value of batch size.')
                raise

            data_size_1 = batch_size if index + batch_size < data_size else data_size - index
            for feature, h in zip(features, hs):
                feature[index:index + data_size_1] = h.d[:data_size_1]
            index += batch_size
    progress(None)

    # Create anomary detection model
    logger.log(99, 'Creating anomary detection model...')
    h = F.concatenate(hs[0], F.unpooling(hs[1], (2, 2)),
                      F.unpooling(hs[2], (4, 4)), axis=1)
    feature_dim = h.shape[1]
    if args.gaussian == 'dense':
        patch_num = features[0].shape[2] * features[0].shape[3]
        mean_param = nn.parameter.get_parameter_or_create(
            name="mean",
            shape=(1, feature_dim, features[0].shape[2], features[0].shape[3]),
            need_grad=False)
        inv_cov_param = nn.parameter.get_parameter_or_create(
            name="cov_inv",
            shape=(feature_dim * patch_num, feature_dim, 1, 1),
            need_grad=False)
        for j in range(features[0].shape[2]):
            for i in range(features[0].shape[3]):
                patch_index = j * features[0].shape[3] + i
                progress(f'{patch_index}/{patch_num}',
                         patch_index * 1.0 / patch_num)
                buf = np.zeros((data_size, feature_dim))
                offset = 0
                for k, feature in enumerate(features):
                    buf[:, offset:offset+feature.shape[1]
                        ] = feature[:, :, j >> k, i >> k]
                    offset += feature.shape[1]
                mean = np.mean(buf, axis=0)
                mean_param.d[:, :, j, i] = mean
                cov = np.cov(buf.T) + 0.01 * np.identity(feature_dim)
                cov_inv = np.linalg.pinv(cov).reshape(
                    feature_dim, feature_dim, 1, 1)
                inv_cov_param.d[feature_dim *
                                patch_index:feature_dim*(patch_index+1)] = cov_inv
        progress(None)

        h2 = h - mean_param
        h3 = F.reshape(F.transpose(h2, (0, 2, 3, 1)),
                       (h2.shape[0], h2.shape[1] * h2.shape[2] * h2.shape[3], 1, 1))
        h3 = F.convolution(h3, weight=inv_cov_param, group=patch_num)
        h3 = F.transpose(F.reshape(
            h3, (h2.shape[0], h2.shape[2], h2.shape[3], h2.shape[1])), (0, 3, 1, 2))
        y = F.pow_scalar(F.mean(h3 * h2, axis=1), 0.5)
    elif args.gaussian == '1/16':
        patch_num = features[2].shape[2] * features[2].shape[3]
        mean_param = nn.parameter.get_parameter_or_create(
            name="mean",
            shape=(1, feature_dim, features[2].shape[2], features[2].shape[3]),
            need_grad=False)
        inv_cov_param = nn.parameter.get_parameter_or_create(
            name="cov_inv",
            shape=(feature_dim * patch_num, feature_dim, 1, 1),
            need_grad=False)
        for j in range(features[2].shape[2]):
            for i in range(features[2].shape[3]):
                patch_index = j * features[2].shape[3] + i
                progress(f'{patch_index}/{patch_num}',
                         patch_index * 1.0 / patch_num)
                buf = np.zeros((data_size, 4, 4, feature_dim))
                offset = 0
                for k, feature in enumerate(features):
                    a = feature[:, :, j << (
                        2-k):(j+1) << (2-k), i << (2-k):(i+1) << (2-k)]
                    a = a.transpose(0, 2, 3, 1)
                    for j2 in range(4):
                        for i2 in range(4):
                            buf[:, j2, i2, offset:offset+a.shape[3]
                                ] = a[:, j2 >> k, i2 >> k, :]
                    offset += a.shape[3]
                buf = buf.reshape(data_size * 4 * 4, feature_dim)
                mean = np.mean(buf, axis=0)
                mean_param.d[:, :, j, i] = mean
                cov = np.cov(buf.T) + 0.01 * np.identity(feature_dim)
                cov_inv = np.linalg.pinv(cov).reshape(
                    feature_dim, feature_dim, 1, 1)
                inv_cov_param.d[feature_dim *
                                patch_index:feature_dim*(patch_index+1)] = cov_inv
        progress(None)

        h2 = h - F.unpooling(mean_param, (4, 4))
        h3 = F.transpose(F.reshape(
            h2, (h2.shape[0], h2.shape[1], h2.shape[2]/4, 4, h2.shape[3]/4, 4)), (0, 2, 4, 1, 3, 5))
        h4 = F.reshape(
            h3, (h3.shape[0], h3.shape[1] * h3.shape[2] * h3.shape[3], h3.shape[4], h3.shape[5]))
        h4 = F.convolution(h4, weight=inv_cov_param, group=patch_num)
        h4 = F.transpose(F.reshape(
            h4, (h2.shape[0], h2.shape[2]/4, h2.shape[3]/4, h2.shape[1], 4, 4)), (0, 3, 1, 4, 2, 5))
        h4 = F.reshape(h4, h2.shape)
        y = F.pow_scalar(F.sum(h4 * h2, axis=1), 0.5)
    elif args.gaussian == 'single':
        buf = np.zeros(
            (data_size, feature_dim, features[0].shape[2], features[0].shape[3]))
        offset = 0
        for k, feature in enumerate(features):
            for j2 in range(buf.shape[2]):
                for i2 in range(buf.shape[3]):
                    buf[:, offset:offset+feature.shape[1], j2,
                        i2] = feature[:, :, j2 >> k, i2 >> k]
            offset += feature.shape[1]
        buf = buf.transpose(0, 2, 3, 1)
        buf = buf.reshape(data_size * buf.shape[1] * buf.shape[2], feature_dim)
        mean = np.mean(buf, axis=0).reshape(1, feature_dim, 1, 1)
        cov = np.cov(buf.T) + 0.01 * np.identity(feature_dim)
        cov_inv = np.linalg.pinv(cov).reshape(feature_dim, feature_dim, 1, 1)
        mean_param = nn.parameter.get_parameter_or_create(
            name="mean",
            shape=mean.shape,
            need_grad=False,
            initializer=mean)
        h2 = h - mean_param
        cov_param = nn.parameter.get_parameter_or_create(
            name="cov_inv",
            shape=cov_inv.shape,
            need_grad=False,
            initializer=cov_inv)
        h3 = F.convolution(h2, weight=cov_param)
        y = F.pow_scalar(F.sum(h3 * h2, axis=1), 0.5)
    else:
        logger.critical(f'{args.gaussian} is not supported.')
        raise

    # Create anomary detection model
    logger.log(99, 'Saving anomary detection model...')
    contents = {
        'global_config': {'default_context': get_extension_context('cudnn')},
        'networks': [
            {'name': 'network',
             'batch_size': batch_size,
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
        description='PaDiM (C1)\n' +
        '\n' +
        'PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization\n' +
        'Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier\n' +
        'https://arxiv.org/abs/2011.08785', formatter_class=argparse.RawTextHelpFormatter)
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
        default='resnet50')
    parser.add_argument(
        '-fr',
        '--feature_ratio',
        help='ratio of features used for anomaly detection(float),default=0.1',
        default=0.1,
        type=float)
    parser.add_argument(
        '-g',
        '--gaussian',
        help='number of Gaussian distributions(option:dense,1/16,single),default=dense',
        default='dense')
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
