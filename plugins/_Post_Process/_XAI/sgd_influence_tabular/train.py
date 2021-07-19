# Copyright 2021 Sony Group Corporation.
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
from sgd_influence.utils import get_context, ensure_dir
import os
import sys
import functools
import numpy as np
from tqdm import tqdm
import nnabla as nn
import nnabla.functions as F
from .network import get_solver, select_model, get_n_classes
from .network import get_batch_data, get_indices
from .network import get_batch_indices, get_config, setup_dataset
from .args import get_train_args
par_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
if par_dir not in sys.path:
    sys.path.append(par_dir)


class SaveController:

    def __init__(self, bundle_size, num_steps, weight_dir, base_name):
        self._total_count = 0
        self._count = 0
        self._num_steps = num_steps
        self._bundle_size = bundle_size
        self._weight_dir = weight_dir
        self._base_name = base_name
        self._params_dict = {}

    def save(self, params, step_size):
        self._params_dict[self._total_count] = params
        self._countup()
        self._save(step_size)

    def _countup(self):
        self._count += 1
        self._total_count += 1

    def _reset(self):
        self._count = 0
        self._params_dict = {}

    def _save(self, step):
        if (self._count == self._bundle_size) | (self._num_steps == step + 1):
            for k, v in self._params_dict.items():
                fn = '%s%04d.h5' % (self._base_name, k)
                save_weight(v, self._weight_dir, fn)
            self._reset()


def save_weight(weight, weight_dir, filename):
    ensure_dir(weight_dir)
    nn.save_parameters(
        os.path.join(weight_dir, filename),
        params=weight,
        extension=".h5"
    )


def train(cfg):
    device_id = cfg.device_id
    seed = cfg.seed
    batch_size = cfg.batch_size
    infl_end_epoch = cfg.infl_end_epoch
    lr = cfg.lr
    bsa = cfg.bs_adjuster
    alpha = cfg.alpha
    n_classes = get_n_classes(cfg.train_csv, cfg.label_variable)
    os.environ['NNABLA_CUDNN_DETERMINISTIC'] = '1'
    os.environ['NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC'] = '1'
    # gpu/cpu
    ctx = get_context(device_id)
    nn.set_default_context(ctx)
    # fetch data
    trainset, n_features = setup_dataset(cfg.train_csv)
    n_tr = trainset.size
    idx_train = get_indices(n_tr, seed)
    network = cfg.network
    # Create training graphs
    test = False
    train_model = functools.partial(
        select_model, network=network, n_classes=n_classes, n_features=n_features, net_name_dict=cfg.net_name_dict, test=test)
    # fit
    save_dir = "%s/seed_%02d" % (cfg.temp_dir, seed)
    bundle_size = 200
    base_name = 'model_step'
    model = train_model
    loss_train = None
    solver = get_solver(network, cfg.network_info, lr)
    solver.set_parameters(nn.get_parameters(grad_only=False))
    lr_n = lr
    c = 0
    info_list = []
    for epoch in tqdm(range(cfg.num_epochs), desc='train'):
        info = []
        weight_dir = os.path.join(save_dir, 'epoch%02d' % (epoch), 'weights')
        idx_list = get_batch_indices(n_tr, batch_size, seed=epoch)
        save_controller = SaveController(
            bundle_size, len(idx_list), weight_dir, base_name)
        # save initial model
        params = nn.get_parameters(grad_only=False).copy()
        if epoch >= infl_end_epoch:
            save_weight(params, weight_dir, 'initial_model.h5')
        for step, idx in enumerate(idx_list):
            info.append({'epoch': epoch, 'step': step,
                         'idx': idx, 'lr': lr_n, 'alpha': alpha})
            c += 1
            # store model
            params = nn.get_parameters(grad_only=False).copy()
            if (not cfg.only_last_params) & (epoch >= infl_end_epoch):
                save_controller.save(params, step)
            X, y = get_batch_data(trainset, idx_train, idx)
            _, loss_train, input_data_train = bsa.adjust_batch_size(
                model, len(X), loss_train)
            solver.set_parameters(params)
            solver.set_learning_rate(lr)
            input_data_train["feat"].d = X
            input_data_train["label"].d = y
            for _k, p in nn.get_parameters(grad_only=False).items():
                loss_train += 0.5 * alpha * F.sum(p * p)
            loss_train.forward()
            solver.zero_grad()
            loss_train.backward()
            for _k, p in nn.get_parameters(grad_only=False).items():
                p.g *= len(X) / idx.size
            solver.update()

            # decay
            if cfg.decay:
                decay_rate = np.sqrt(c / (c + 1))
                lr_n *= decay_rate

        params = nn.get_parameters(grad_only=False).copy()
        if epoch >= infl_end_epoch:
            save_weight(params, weight_dir, 'final_model.h5')
        info_list.append(info)
    # save info
    np.save(os.path.join(save_dir, cfg.info_filename), arr=info_list)


if __name__ == '__main__':
    args = get_train_args()
    config = get_config(args)
    train(config)
