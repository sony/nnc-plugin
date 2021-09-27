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

import numpy as np
import nnabla as nn
import nnabla.solvers as S
import nnabla.functions as F
import os
import functools
from tqdm import tqdm
from sgd_influence.model import setup_model
from sgd_influence.dataset import get_batch_data, init_dataset, get_data, get_image_size, get_batch_indices
from sgd_influence.utils import get_indices, save_to_csv, is_proto_graph
from sgd_influence.infl import compute_gradient, save_infl_for_analysis


def infl_icml(model_info_dict, file_dir_dict, use_all_params, need_evaluate, alpha):
    num_epochs = 2
    # params
    lr = model_info_dict['lr']
    seed = model_info_dict['seed']
    net_func = model_info_dict['net_func']
    batch_size = model_info_dict['batch_size']
    target_epoch = model_info_dict['num_epochs']
    # files and dirs
    save_dir = file_dir_dict['save_dir']
    infl_filename = file_dir_dict['infl_filename']
    network_info = model_info_dict['network_info']
    net_name_dict = model_info_dict['net_name_dict']
    bsa = model_info_dict['bs_adjuster']
    final_model_name = file_dir_dict['weight_name_dict']['final']
    final_model_path = os.path.join(save_dir, 'epoch%02d' % (
        target_epoch - 1), 'weights', final_model_name)
    input_dir_name = os.path.dirname(file_dir_dict['train_csv'])
    # setup
    trainset, valset, image_shape, n_classes, ntr, nval = init_dataset(
        file_dir_dict['train_csv'], file_dir_dict['val_csv'], seed)
    n_channels, _h, _w = image_shape

    idx_train = get_indices(ntr, seed)
    idx_val = get_indices(nval, seed)
    if is_proto_graph(net_func):
        resize_size_train = model_info_dict['resize_size_train']
        resize_size_val = model_info_dict['resize_size_val']
        solver = network_info.optimizers['Optimizer'].solver
        _setup_model = functools.partial(
            setup_model, net_name_dict=net_name_dict)
    else:
        resize_size_train = get_image_size((_h, _w))
        resize_size_val = resize_size_train
        solver = S.Momentum(lr=lr, momentum=0.9)
        _setup_model = setup_model
    nn.load_parameters(final_model_path)
    trained_params = nn.get_parameters(grad_only=False)
    test = True
    grad_model = functools.partial(
        _setup_model, network=net_func, n_classes=n_classes, n_channels=n_channels, resize_size=resize_size_val, test=test, reduction='sum')

    solver.set_parameters(trained_params)
    # gradient
    u = compute_gradient(grad_model, bsa, solver, valset,
                         batch_size, idx_val, resize_size_val)

    # Hinv * u with SGD
    seed_train = 0
    v = {
        f'{k}': nn.Variable(p.d.shape, need_grad=True)
        for k, p in nn.get_parameters(grad_only=False).items()
    }
    for k, vv in v.items():
        vv.d = 0

    solver.set_parameters(v)
    loss_train = []
    loss_fn = None
    test = False
    for epoch in tqdm(range(num_epochs)):
        # training
        seed_train = 0
        np.random.seed(epoch)
        idx = get_batch_indices(ntr, batch_size, seed=epoch)
        for j, i in enumerate(idx):
            seeds = list(range(seed_train, seed_train + i.size))
            seed_train += i.size
            X, y = get_batch_data(trainset, idx_train, i,
                                  resize_size_train, test=test, seeds=seeds)
            _, loss_fn, input_image = bsa.adjust_batch_size(
                grad_model, len(X), loss_fn, test=test)
            input_image["image"].d = X
            input_image["label"].d = y
            loss_fn.forward()

            vg = 0
            params = nn.get_parameters(grad_only=False)
            for k, gp in zip(params.keys(), nn.grad([loss_fn], params.values())):
                # for k, g in grad_params.items():
                vv = v.get(k, None)
                if vv is not None:
                    vg += F.sum(vv * gp)
            for k, p in nn.get_parameters(grad_only=False).items():
                p.grad.zero()

            loss_i = 0
            params = nn.get_parameters(grad_only=False)
            for k, vgp in zip(params.keys(), nn.grad([vg], params.values())):
                vv = v.get(k, None)
                uu = u.get(k, None)
                if (vv is not None) & (uu is not None):
                    loss_i += 0.5 * \
                        F.sum(vgp * vv + alpha * vv * vv) - F.sum(uu * vv)
            loss_i.forward()
            solver.zero_grad()
            loss_i.backward(clear_buffer=True)
            solver.update()
            loss_train.append(loss_i.d.copy())

    # influence
    infl_dict = dict()
    infl = np.zeros(ntr)
    for i in tqdm(range(ntr), desc='calc influence (3/3 steps)'):
        csv_idx = idx_train[i]
        file_name = trainset.get_filepath_to_data(csv_idx)
        file_name = os.path.join(input_dir_name, file_name)
        file_name = os.path.normpath(file_name)
        X, y = get_data(
            trainset, idx_train[i], resize_size_train, True, seed=i)
        _, loss_fn, input_image = bsa.adjust_batch_size(
            grad_model, len(X), loss_fn, test=test)
        input_image["image"].d = X
        input_image["label"].d = y
        loss_fn.forward()
        for p in nn.get_parameters(grad_only=False).values():
            p.grad.zero()
        loss_fn.backward(clear_buffer=True)
        infl_i = 0
        for k, p in nn.get_parameters(grad_only=False).items():
            vv = v[k]
            if vv is not None:
                infl_i += (p.g * vv.d).sum()
        infl[i] = -infl_i / ntr
        infl_dict[csv_idx] = [file_name, y, infl[i]]
    infl_list = [val + [key] for key, val in infl_dict.items()]
    infl_list = sorted(infl_list, key=lambda x: (x[-2]))

    # save
    header = ['x:image', 'y:label', 'influence', 'datasource_index']
    data_type = 'object,int,float,int'
    if need_evaluate:
        save_infl_for_analysis(infl_list, use_all_params,
                               save_dir, infl_filename, epoch, header, data_type)
    save_to_csv(filename=infl_filename, header=header,
                list_to_save=infl_list, data_type=data_type)
