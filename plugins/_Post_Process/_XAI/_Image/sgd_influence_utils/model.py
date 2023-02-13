# Copyright 2023 Sony Group Corporation.
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
import functools
import numpy as np
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
from nnabla.core.graph_def import ProtoGraph
from nnabla.utils.nnp_graph import NnpNetwork
from nnabla.utils.load import load
from .utils import is_proto_graph


def block(x, scope_name, n_channels, kernel, pad, test):
    with nn.parameter_scope(scope_name):
        with nn.parameter_scope('conv1'):
            h = PF.convolution(x, n_channels, kernel=kernel,
                               pad=pad, with_bias=True)
            h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)

        with nn.parameter_scope('conv2'):
            h = PF.convolution(h, n_channels, kernel=kernel,
                               pad=pad, with_bias=True)
            h = F.relu(h)
            h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
    return h


def cifarnet(x, test=False, n_classes=10):
    maps = [32, 64, 128]
    kernel = (3, 3)
    pad = (1, 1)

    h = block(x, 'block1', maps[0], kernel, pad, test)
    h = block(h, 'block2', maps[1], kernel, pad, test)
    h = block(h, 'block3', maps[2], kernel, pad, test)
    h = PF.affine(h, n_classes)
    return h


def calc_loss_reduction(loss, reduction='mean'):
    loss_dict = {
        'mean': F.mean(loss),
        'sum': F.sum(loss)
    }
    return loss_dict[reduction]


def calc_acc(pred, label, method='mean'):
    acc_sum = (np.argmax(pred, axis=1).reshape(-1, 1)
               == np.array(label).reshape(-1, 1)).sum()
    acc_dict = {
        'mean': acc_sum / len(label),
        'sum': acc_sum
    }
    return acc_dict[method]


def get_network(proto_graph, name, batch_size):
    return NnpNetwork(proto_graph[name], None, None)


def setup_model(**kwargs):
    if is_proto_graph(kwargs['network']):
        return setup_model_nnp(**kwargs)
    else:
        return setup_model_func(**kwargs)


def setup_model_nnp(network, n_classes, n_channels, resize_size, batch_size, test, net_name_dict, reduction='mean'):
    if test:
        _network = get_network(
            network, net_name_dict['validation'], batch_size=batch_size)
    else:
        _network = get_network(
            network, net_name_dict['train'], batch_size=batch_size)
    image = list(_network.inputs.values())[0]
    label = list(_network.inputs.values())[-1]
    pred = [v for (k, v) in _network.variables.items() if '/' not in k][-2]
    loss = list(_network.outputs.values())[-1]
    loss = calc_loss_reduction(loss, reduction)
    input_image = {"image": image, "label": label}
    return pred, loss, input_image


def get_input_size(net_func, net_name, batch_size):
    network = get_network(net_func, net_name, batch_size)
    input_variable = list(network.inputs.values())[0]
    input_shape = (input_variable.shape[2], input_variable.shape[3])
    return input_shape


def setup_model_func(network, n_classes, n_channels, resize_size, batch_size, test, reduction='mean'):
    prediction = functools.partial(network, n_classes=n_classes)
    image = nn.Variable(
        (batch_size, n_channels, resize_size[0], resize_size[1]))
    label = nn.Variable((batch_size, 1))
    pred = prediction(image, test)
    loss = F.softmax_cross_entropy(pred, label)
    loss = calc_loss_reduction(loss, reduction)
    input_image = {"image": image, "label": label}
    return pred, loss, input_image


def _adjust_batch_size(model, batch_size, loss_fn=None, test=False):
    has_loss = loss_fn is not None
    if has_loss:
        loss_d, loss_g = loss_fn.d, loss_fn.g
    pred, loss_fn, input_image = model(batch_size=batch_size, test=test)
    if has_loss:
        loss_fn.d = loss_d
        loss_fn.g = loss_g
    return pred, loss_fn, input_image


class BatchSizeAdjuster:

    def __init__(self, proto, batch_size):
        self._proto = proto
        self._get_network(batch_size)
        self.pre_batch_size = None
        self.test = None

    def adjust_batch_size(self, model, batch_size, loss=None, test=False):
        if (self.pre_batch_size != batch_size) | (self.test != test):
            self._get_network(batch_size)
            model = functools.partial(
                model, network=self.network, batch_size=batch_size)
            self._pred, self._loss, self._input = _adjust_batch_size(
                model, batch_size, loss, test)
        self.pre_batch_size = batch_size
        self.test = test
        return self._pred, self._loss, self._input

    def _get_network(self, batch_size):
        self.network = None
        rng = np.random.RandomState(1223)
        self.network = ProtoGraph.from_proto(
            proto=self._proto, batch_size=batch_size, param_scope=None, rng=rng)


def get_config(args, is_eval=False):
    class Congig(object):
        def __init__(self, cfg, is_eval):
            lr = 0.05

            self.model_info_dict = {
                'lr': lr,
                'batch_size': cfg.batch_size,
                'device_id': cfg.device_id,
                # 'seed': cfg.seed,
                'net_func': cifarnet,
            }

            self.file_dir_dict = {
                'train_csv': cfg.input_train,
                'val_csv': cfg.input_val,
                'score_filename': cfg.score_output,
                'save_dir': cfg.weight_output,
                'infl_filename': cfg.output,
                'weight_name_dict': {'final': 'final_model.h5', 'initial': 'initial_model.h5'},
                'info_filename': 'info.npy'
            }
            self._select_model(cfg)
            if is_eval:
                self._init_for_eval(cfg)
            else:
                self._init_for_train_infl(cfg)

        def _init_for_eval(self, cfg):
            self.file_dir_dict['test_csv'] = cfg.input_test
            start_epoch = 0 if cfg.retrain_all else self.model_info_dict[
                'num_epochs'] - 1
            self.model_info_dict['start_epoch'] = start_epoch

        def _init_for_train_infl(self, cfg):
            self.calc_infl_with_all_params = cfg.calc_infl_with_all_params
            self.need_evaluate = False if cfg.score_output is None else True
            end_epoch_dict = {
                'last': self.model_info_dict['num_epochs'] - 1, 'all': 0}
            self.model_info_dict['end_epoch'] = end_epoch_dict[
                cfg.calc_infl_method]

        def _select_model(self, cfg):
            def has_network_name(nnp, net_name):
                return net_name in nnp.get_network_names()

            if args.model:
                info = load([cfg.model], prepare_data_iterator=False)
                self.model_info_dict['bs_adjuster'] = BatchSizeAdjuster(
                    info.proto, cfg.batch_size)
                net = self.model_info_dict['bs_adjuster'].network
                net_name_dict = {
                    'train': info.proto.optimizer.pop().network_name,
                    'validation': info.proto.monitor.pop().network_name,
                }
                self.model_info_dict['resize_size_train'] = get_input_size(
                    net, net_name_dict['train'], cfg.batch_size)
                self.model_info_dict['resize_size_val'] = get_input_size(
                    net, net_name_dict['validation'], cfg.batch_size)
                self.model_info_dict['net_name_dict'] = net_name_dict
                self.model_info_dict['num_epochs'] = 1
            else:
                net = cifarnet
                info = None
                self.model_info_dict['num_epochs'] = cfg.n_epochs
                self.model_info_dict['net_name_dict'] = None
            self.model_info_dict['net_func'] = net
            self.model_info_dict['network_info'] = info

    return Congig(args, is_eval)
