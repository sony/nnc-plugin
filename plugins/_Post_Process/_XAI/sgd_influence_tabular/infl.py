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

import os
import functools
import numpy as np
from tqdm import tqdm
import nnabla as nn
import nnabla.functions as F
from sgd_influence_tabular.network import get_solver, select_model, get_n_classes
from sgd_influence_tabular.network import get_indices, get_batch_indices, get_batch_data
from sgd_influence_tabular.network import get_config, setup_dataset
from sgd_influence_tabular.args import get_infl_args
from utils.model import get_context
from utils.file import save_to_csv


def get_weight_name(save_dir, epoch, step, only_last_params, final_model_name='final_model.h5'):
    base_name = '%s/epoch%02d/weights' % (save_dir, epoch)
    if only_last_params:
        fn = os.path.join(base_name, final_model_name)
    else:
        fn = os.path.join(base_name, 'model_step%04d.h5' % (step))
    return fn


def compute_gradient(bs_adjuster, grad_model, solver, dataset, batch_size, idx_list_to_data, alpha):
    n = len(idx_list_to_data)
    grad_idx = get_batch_indices(n, batch_size, seed=None)
    u = {}
    loss = None
    for idx in grad_idx:
        X, y = get_batch_data(dataset, idx_list_to_data, idx)
        _, loss, input_data_train = bs_adjuster.adjust_batch_size(
            grad_model, len(idx), loss)
        input_data_train["feat"].d = X
        input_data_train["label"].d = y
        loss.forward()
        for k, p in nn.get_parameters().items():
            p.grad.zero()
        loss.backward()

    for k, p in nn.get_parameters(grad_only=False).items():
        key = f'{k}_u'
        uu = u.get(key, None)
        if uu is None:
            u[key] = nn.Variable(p.shape)
            u[key].data.zero()
            u[key].grad.zero()
        u[key].d = p.g.copy()
    return u


def infl_sgd(cfg):
    device_id = cfg.device_id
    seed = cfg.seed
    batch_size = cfg.batch_size
    bsa = cfg.bs_adjuster
    lr = cfg.lr
    n_classes = get_n_classes(cfg.train_csv, cfg.label_variable)
    os.environ['NNABLA_CUDNN_DETERMINISTIC'] = '1'
    os.environ['NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC'] = '1'
    # gpu/cpu
    ctx = get_context(device_id)
    nn.set_default_context(ctx)
    # fetch data
    trainset, n_features = setup_dataset(cfg.train_csv)
    valset, _ = setup_dataset(cfg.val_csv)
    n_tr = trainset.size
    n_val = valset.size
    idx_train = get_indices(n_tr, seed)
    infl_filename = cfg.infl_filepath
    save_dir = '%s/seed_%02d' % (cfg.temp_dir, seed)
    info_filename = 'info.npy'
    network = cfg.network
    _setup_model = functools.partial(
        select_model, network=network, n_classes=n_classes, n_features=n_features, net_name_dict=cfg.net_name_dict)
    # Create validation graphs
    test = True
    grad_model = functools.partial(_setup_model, test=test)
    # Create training graphs
    test = False
    infl_model = functools.partial(_setup_model, test=test)
    # model setup
    info_list = np.load(os.path.join(
        save_dir, info_filename), allow_pickle=True)
    solver = get_solver(network, cfg.network_info, lr)
    fn = '%s/epoch%02d/weights/final_model.h5' % (save_dir, len(info_list)-1)
    nn.load_parameters(fn)
    trained_params = nn.get_parameters()
    lr = info_list[-1][-1]['lr']

    solver.set_parameters(trained_params, reset=True)
    # gradient
    idx_train = get_indices(n_tr, seed)
    idx_val = get_indices(n_val, seed)
    u = compute_gradient(bsa, grad_model, solver, valset,
                         batch_size, idx_val, cfg.alpha)
    loss = None
    infl_dict = {}
    for epoch in range(cfg.infl_end_epoch, cfg.num_epochs)[::-1]:
        for step_info in tqdm(info_list[epoch][::-1], desc='calculate influence'):
            step, idx, lr, alpha = step_info['step'], step_info['idx'], step_info['lr'], step_info['alpha']
            fn = get_weight_name(save_dir, epoch, step, cfg.only_last_params)
            nn.load_parameters(fn)
            params = nn.get_parameters()
            _, loss, input_data = bsa.adjust_batch_size(infl_model, 1, loss)
            solver.set_parameters(params)
            solver.set_learning_rate(lr)
            for i in idx:
                X, y = get_batch_data(trainset, idx_train, [i])
                input_data["feat"].d = X
                input_data["label"].d = y
                for _k, p in nn.get_parameters().items():
                    loss += 0.5 * alpha * F.sum(p * p)
                loss.forward()
                for k, p in nn.get_parameters().items():
                    p.grad.zero()
                loss.backward()
                csv_idx = idx_train[i]
                infl = infl_dict.get(csv_idx, 0.0)
                for k, p in nn.get_parameters().items():
                    infl += lr * (u[f'{k}_u'].d * p.g).sum().item() / idx.size
                infl_dict[csv_idx] = infl
            # update u
            params = nn.get_parameters()
            X, y = get_batch_data(trainset, idx_train, idx)
            _, loss, input_data = bsa.adjust_batch_size(
                infl_model, len(X), loss)
            solver.set_parameters(params, reset=True)
            input_data["feat"].d = X
            input_data["label"].d = y

            for _k, p in nn.get_parameters().items():
                loss += 0.5 * alpha * F.sum(p * p)

            loss.forward()
            params = nn.get_parameters()
            ug = 0
            for k, _gp in zip(params.keys(), nn.grad([loss], params.values())):
                gp = _gp.get_unlinked_variable()
                uu = u.get(f'{k}_u', None)
                if uu is not None:
                    ug += F.sum(uu * gp)
            ug.forward()
            for k, p in nn.get_parameters().items():
                p.grad.zero()
            ug.backward()
            for k, p in nn.get_parameters().items():
                key = f'{k}_u'
                u[key].d -= lr * p.g / len(X)
    # save
    # sort by influence score
    infl_list = [[k] + [v] for k, v in infl_dict.items()]
    infl_list = sorted(infl_list, key=lambda x: (x[-1]))
    data_type = 'int,float'
    header = ['index', 'influence']
    save_to_csv(filename=infl_filename, header=header,
                list_to_save=infl_list, data_type=data_type)


if __name__ == '__main__':
    args = get_infl_args()
    config = get_config(args)
    infl_sgd(config)
