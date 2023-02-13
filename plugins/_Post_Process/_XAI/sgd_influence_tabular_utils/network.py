# Copyright 2021,2022 Sony Group Corporation.
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

import numpy as np
import functools
import nnabla as nn
import nnabla.solvers as S
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.load import load
from nnabla.utils.nnp_graph import NnpLoader
from nnabla.utils.data_source_implements import CsvDataSource


def block(x, scope_name, n_channels):
    with nn.parameter_scope(scope_name):
        with nn.parameter_scope('layer1'):
            h = PF.affine(x, n_channels)
            h = F.relu(h)
    return h


def logreg(x, n_classes=1):
    return PF.affine(x, n_classes)


def dnn(x, m=[8, 8], n_classes=1):
    h = block(x, 'block1', m[0])
    h = F.dropout(h, output_mask=True)[0]
    h = block(h, 'block2', m[1])
    h = PF.affine(h, n_classes)
    return h


def calc_acc(pred, label, method='mean'):
    acc_sum = (np.argmax(pred, axis=1).reshape(-1, 1)
               == np.array(label).reshape(-1, 1)).sum()
    acc_dict = {
        'mean': acc_sum / len(label),
        'sum': acc_sum
    }
    return acc_dict[method]


def calc_loss_reduction(loss, reduction='mean'):
    loss_dict = {
        'mean': F.mean,
        'sum': F.sum,
    }
    return loss_dict[reduction](loss)


def is_nnp(network):
    return isinstance(network, nn.utils.nnp_graph.NnpLoader)


def get_solver(network, network_info, lr):
    if is_nnp(network):
        solver = network_info.optimizers['Optimizer'].solver
    else:
        solver = S.Sgd(lr=lr)
    return solver


def select_model(network, **kwargs):
    if is_nnp(network):
        return setup_model_nnp(network, **kwargs)
    else:
        return setup_model_func(network, **kwargs)


def setup_model_nnp(network, n_classes, n_features, batch_size, test, net_name_dict, reduction='mean'):
    if test:
        net_name = net_name_dict['validation']
    else:
        net_name = net_name_dict['train']
    _network = network.get_network(net_name, batch_size=batch_size)
    feat = list(_network.inputs.values())[0]
    label = list(_network.inputs.values())[-1]
    pred = [v for (k, v) in _network.variables.items() if '/' not in k][-2]
    loss = list(_network.outputs.values())[-1]
    loss = calc_loss_reduction(loss, reduction)
    input_data = {"feat": feat, "label": label}
    return pred, loss, input_data


def setup_model_func(network, n_classes, n_features, batch_size, test, net_name_dict, reduction='mean', is_regression=False):
    prediction = network
    feat = nn.Variable((batch_size, n_features))
    target = nn.Variable((batch_size, 1))
    pred = prediction(feat)
    if is_regression:
        loss = F.mean(F.pow_scalar(F.sub2(pred, target), 2))
    else:
        loss = F.sigmoid_cross_entropy(pred, target)
    loss = calc_loss_reduction(loss, reduction)
    input_data = {"feat": feat, "label": target}
    return pred, loss, input_data


def setup_dataset(filename):
    dataset = CsvDataSource(
        filename=filename,
        shuffle=False,
        normalize=True
    )
    _x, _y = get_data(dataset, 0)
    n_features = np.array(_x).shape[-1]
    return dataset, n_features


def get_batch_data(dataset, idx_list_to_data, idx_list_to_idx, escape_list=[]):
    X = []
    y = []
    for idx_to_idx in idx_list_to_idx:
        idx_to_data = idx_list_to_data[idx_to_idx]
        if idx_to_data in escape_list:
            continue
        _x, _y = get_data(dataset, idx_to_data)
        X.append(_x)
        y.append(_y)
    y = np.array(y).reshape(-1, 1)
    return X, y


def get_data(dataset, idx):
    x, y = dataset._get_data(idx)
    return x, y


def _adjust_batch_size(model, batch_size, loss=None, test=False):
    has_loss = loss is not None
    if has_loss:
        loss_d, loss_g = loss.d, loss.g
    pred, loss, input_data = model(batch_size=batch_size, test=test)
    if has_loss:
        loss.d = loss_d
        loss.g = loss_g
    return pred, loss, input_data


def get_batch_indices(num_data, batch_size, seed=None):
    if seed is None:
        shuffled_idx = np.arange(num_data)
    else:
        shuffled_idx = get_indices(num_data, seed)
    indices_list = []
    for i in range(0, num_data, batch_size):
        indices = shuffled_idx[i:i+batch_size]
        indices_list.append(indices)
    return indices_list


def get_indices(n, seed):
    np.random.seed(seed)
    idx = np.random.permutation(n)
    return idx


def get_n_classes(input_csv_path, label_variable):
    csv_file = np.loadtxt(input_csv_path, delimiter=",", dtype=str)
    header = [item.split(":")[0] for item in csv_file[0]]
    label_column = [s for s in header if s[0] == label_variable][0]
    labels = csv_file[1:, header.index(label_column)]
    num_classes = np.unique(labels).size
    return num_classes


class BatchSizeAdjuster:
    def __init__(self, filepath):
        self._nnp_filepath = filepath
        self._nnp_loader = NnpLoader
        self._read_nnp_network()
        self._cnt = 0
        self._pre_bs = None
        self._pred = None
        self._loss = None
        self._inpt = None

    def adjust_batch_size(self, model, batch_size, loss=None, test=False):
        if self._pre_bs == batch_size:
            return self._pred, self._loss, self._inpt
        else:
            self._read_nnp_network()
            model = functools.partial(model, network=self.nnp_network)
            self._pred, self._loss, self._inpt = _adjust_batch_size(
                model, batch_size, loss, test)
            return self._pred, self._loss, self._inpt

    def _read_nnp_network(self):
        self.nnp_network = None
        self.nnp_network = self._nnp_loader(self._nnp_filepath)


class BatchSizeAdjusterFunc:
    def __init__(self):
        self._pre_bs = None
        self._pred = None
        self._loss = None
        self._inpt = None

    def adjust_batch_size(self, model, batch_size, loss=None, test=False):
        if self._pre_bs == batch_size:
            return self._pred, self._loss, self._inpt
        else:
            self._pred, self._loss, self._inpt = _adjust_batch_size(
                model, batch_size, loss, test)
            return self._pred, self._loss, self._inpt


class Config:
    def __init__(self, args, is_eval):
        self.temp_dir = args.temp_dir
        self.target = args.target
        self.model = args.model
        self.device_id = args.device_id
        self.retrain_all = args.retrain_all
        self.lr = 0.001
        self.decay = True
        self.batch_size = args.batch_size
        self.m = [8, 8]
        self.alpha = 0.001
        self.info_filename = 'info.npy'
        self._select_model(args)
        self.only_last_params = args.only_last_params
        self.set_retrain_config(args.retrain_all)
        self.label_variable = args.label_variable
        if is_eval:
            self._init_for_eval(args)
        else:
            self._init_for_train_infl(args)

    def _init_for_eval(self, args):
        self.n_trials = args.n_trials
        self.remove_n_list = args.remove_n_list
        self.output_dir = args.output_dir
        self.dataset = args.dataset
        self.need_evaluate = True

    def _init_for_train_infl(self, args):
        self.train_csv = args.input_train
        self.val_csv = args.input_val
        self.infl_filepath = args.output

    def set_retrain_config(self, retrain_all):
        self.retrain_all = retrain_all
        self.infl_end_epoch = 0 if retrain_all else self.num_epochs - 1
        self.eval_start_epoch = self.infl_end_epoch

    def _select_model(self, args):
        def has_network_name(nnp, net_name):
            return net_name in nnp.get_network_names()

        if args.model:
            self.bs_adjuster = BatchSizeAdjuster(args.model)
            net = self.bs_adjuster.nnp_network
            info = load([args.model], prepare_data_iterator=False)
            net_name_dict = {
                'train': info.proto.optimizer.pop().network_name,
                'validation': info.proto.monitor.pop().network_name,
            }

            self.net_name_dict = net_name_dict
            if not has_network_name(net, net_name_dict['train']):
                raise KeyError(
                    f"'{net_name_dict['train']}' is not in network names {net.get_network_names()}")
            if not has_network_name(net, net_name_dict['validation']):
                raise KeyError(
                    f"'{net_name_dict['validation']}' is not in network names {net.get_network_names()}")
            self.num_epochs = 1
        else:
            self.bs_adjuster = BatchSizeAdjusterFunc()
            net = dnn
            info = None
            self.num_epochs = args.num_epochs
            self.net_name_dict = None
        self.network = net
        self.network_info = info


def get_config(args, is_eval=False):
    return Config(args, is_eval)
