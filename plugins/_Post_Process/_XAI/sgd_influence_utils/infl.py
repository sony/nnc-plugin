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
import nnabla.solvers as S
import nnabla.functions as F
from .model import setup_model
from .dataset import get_batch_data, init_dataset, get_data, get_image_size, get_batch_indices
from .utils import ensure_dir, get_indices, save_to_csv, is_proto_graph


def select_modelfile_for_infl(use_all_params, final_model_name, save_dir, epoch, step):
    weights_dir = 'weights'
    base_name = '%s/epoch%02d/%s/' % (save_dir, epoch, weights_dir)
    if use_all_params:
        fn = os.path.join(base_name, 'model_step%04d.h5' % (step))
    else:
        fn = os.path.join(base_name, final_model_name)
    return fn


def save_infl_for_analysis(infl_list, use_all_params, save_dir, infl_filename, epoch, header, data_type):
    dn = os.path.join(save_dir, 'epoch%02d' % (epoch))
    if use_all_params:
        dn = os.path.join(dn, 'infl_original')
    else:
        dn = os.path.join(dn, 'infl_arranged')
    ensure_dir(dn)
    save_to_csv(filename=os.path.join(dn, os.path.basename(infl_filename)),
                header=header, list_to_save=infl_list, data_type=data_type)


def infl_sgd(model_info_dict, file_dir_dict, use_all_params, need_evaluate):
    # params
    lr = model_info_dict['lr']
    seed = model_info_dict['seed']
    net_func = model_info_dict['net_func']
    batch_size = model_info_dict['batch_size']
    end_epoch = model_info_dict['end_epoch']
    target_epoch = model_info_dict['num_epochs']
    network_info = model_info_dict['network_info']
    net_name_dict = model_info_dict['net_name_dict']
    bsa = model_info_dict['bs_adjuster']
    # files and dirs
    save_dir = file_dir_dict['save_dir']
    info_filename = file_dir_dict['info_filename']
    infl_filename = file_dir_dict['infl_filename']
    weight_name_dict = file_dir_dict['weight_name_dict']
    input_dir_name = os.path.dirname(file_dir_dict['train_csv'])

    # setup
    trainset, valset, image_shape, n_classes, ntr, nval = init_dataset(
        file_dir_dict['train_csv'], file_dir_dict['val_csv'], seed)
    n_channels, _h, _w = image_shape
    if is_proto_graph(net_func):
        resize_size_train = model_info_dict['resize_size_train']
        resize_size_val = model_info_dict['resize_size_val']
        solver = network_info.optimizers['Optimizer'].solver
        _setup_model = functools.partial(
            setup_model, net_name_dict=net_name_dict)
    else:
        resize_size_train = get_image_size((_h, _w))
        resize_size_val = resize_size_train
        solver = S.Sgd(lr=lr)
        _setup_model = setup_model
    idx_train = get_indices(ntr, seed)
    idx_val = get_indices(nval, seed)

    _select_modelfile_for_infl = functools.partial(
        select_modelfile_for_infl, final_model_name=weight_name_dict['final'], save_dir=save_dir)
    final_model_path = _select_modelfile_for_infl(
        use_all_params=False, epoch=target_epoch - 1, step=None)
    nn.load_parameters(final_model_path)
    trained_params = nn.get_parameters(grad_only=False)
    test = True

    grad_model = functools.partial(
        _setup_model, network=net_func, n_classes=n_classes, n_channels=n_channels, resize_size=resize_size_val, test=test, reduction='sum')

    solver.set_parameters(trained_params)
    # gradient
    u = compute_gradient(grad_model, bsa, solver, valset,
                         batch_size, idx_val, resize_size_val)

    test = False
    infl_model = functools.partial(
        _setup_model, network=net_func, n_classes=n_classes, n_channels=n_channels, resize_size=resize_size_train, test=test)
    # influence
    infl_dict = {}
    info = np.load(os.path.join(save_dir, info_filename), allow_pickle=True)
    loss_fn = None
    for epoch in tqdm(range(target_epoch - 1, end_epoch - 1, -1), desc='calc influence (3/3 steps)'):
        for step_info in info[epoch][::-1]:
            idx, seeds, lr, step = step_info['idx'], step_info[
                'seeds'], step_info['lr'], step_info['step']
            fn = _select_modelfile_for_infl(
                use_all_params=use_all_params, epoch=epoch, step=step)
            _, loss_fn, input_image = bsa.adjust_batch_size(
                infl_model, 1, loss_fn, test)
            nn.load_parameters(fn)
            params = nn.get_parameters(grad_only=False)
            solver.set_parameters(params)
            X = []
            y = []
            for i, seed in zip(idx, seeds):
                i = int(i)
                image, label = get_data(
                    trainset, idx_train[i], resize_size_train, test, seed=seed)
                X.append(image)
                y.append(label)
                input_image["image"].d = image
                input_image["label"].d = label
                loss_fn.forward()
                solver.zero_grad()
                loss_fn.backward(clear_buffer=True)

                csv_idx = idx_train[i]
                infl = infl_dict.get(csv_idx, [0.0])[-1]
                for j, (key, param) in enumerate(nn.get_parameters(grad_only=False).items()):
                    infl += lr * (u[key].d * param.g).sum() / idx.size

                # store infl
                file_name = trainset.get_filepath_to_data(csv_idx)
                file_name = os.path.join(input_dir_name, file_name)
                file_name = os.path.normpath(file_name)
                infl_dict[csv_idx] = [file_name, label, infl]

            # update u
            _, loss_fn, input_image = bsa.adjust_batch_size(
                infl_model, len(idx), loss_fn, test)
            input_image["image"].d = X
            input_image["label"].d = np.array(y).reshape(-1, 1)
            loss_fn.forward()
            params = nn.get_parameters(grad_only=False)
            grad_params = {}
            for key, p in zip(params.keys(), nn.grad([loss_fn], params.values())):
                if p:
                    grad_params[key] = p.get_unlinked_variable()
            ug = 0
            # compute H[t]u[t]
            for key, uu in u.items():
                gp = grad_params.get(key, None)
                if gp:
                    ug += F.sum(uu * gp)
            ug.forward()
            solver.zero_grad()
            ug.backward(clear_buffer=True)

            for j, (key, param) in enumerate(nn.get_parameters(grad_only=False).items()):
                u[key].d -= lr * param.g / idx.size

        # sort by influence score
        infl_list = [val + [key] for key, val in infl_dict.items()]
        infl_list = sorted(infl_list, key=lambda x: (x[-2]))

        # save
        header = ['x:image', 'y:label', 'influence', 'datasource_index']
        data_type = 'object,int,float,int'
        if need_evaluate:
            save_infl_for_analysis(
                infl_list, use_all_params, save_dir, infl_filename, epoch, header, data_type)
    save_to_csv(filename=infl_filename, header=header,
                list_to_save=infl_list, data_type=data_type)


def compute_gradient(grad_model, bs_adjuster, solver, dataset, batch_size, idx_list_to_data, resize_size):
    n = len(idx_list_to_data)
    grad_idx = get_batch_indices(n, batch_size, seed=None)
    u = {}
    loss_fn = None
    test = True
    for i in tqdm(grad_idx, desc='calc gradient (2/3 steps)'):
        X, y = get_batch_data(dataset, idx_list_to_data,
                              i, resize_size, test=test)
        _, loss_fn, input_image = bs_adjuster.adjust_batch_size(
            grad_model, len(X), loss_fn, test)
        input_image["image"].d = X
        input_image["label"].d = y
        loss_fn.forward()
        solver.zero_grad()
        loss_fn.backward(clear_buffer=True)

        for key, param in nn.get_parameters(grad_only=False).items():
            uu = u.get(key, None)
            if uu is None:
                u[key] = nn.Variable(param.shape)
                u[key].data.zero()
            u[key].d += param.g / n
    return u
