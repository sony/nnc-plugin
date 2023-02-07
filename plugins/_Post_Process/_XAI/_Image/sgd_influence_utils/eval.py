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
import os
import functools
from tqdm import tqdm
import numpy as np
import nnabla as nn
import nnabla.solvers as S
from .model import setup_model
from .utils import get_context, get_indices, save_to_csv, is_proto_graph, read_csv
from .dataset import get_batch_data, init_dataset, get_image_size, setup_nnabla_dataset, get_input_size
from .train import eval_model


def select_model_file(output_path, start_epoch):
    fn = '%s/epoch%02d/weights/initial_model.h5' % (output_path, start_epoch)
    return fn


def retrain(model_info_dict, file_dir_dict, escape_list=[]):
    # params
    lr = model_info_dict['lr']
    seed = model_info_dict['seed']
    net_func = model_info_dict['net_func']
    batch_size = model_info_dict['batch_size']
    end_epoch = model_info_dict['num_epochs']
    start_epoch = model_info_dict['start_epoch']
    network_info = model_info_dict['network_info']
    net_name_dict = model_info_dict['net_name_dict']
    bsa = model_info_dict['bs_adjuster']
    # files and dirs
    save_dir = file_dir_dict['save_dir']
    info_filename = file_dir_dict['info_filename']
    score_filename = file_dir_dict['score_filename']
    info_file_path = os.path.join(save_dir, info_filename)
    # setup
    trainset, valset, image_shape, n_classes, ntr, nval = init_dataset(
        file_dir_dict['train_csv'], file_dir_dict['val_csv'], seed)
    testset = setup_nnabla_dataset(file_dir_dict['test_csv'])
    ntest = testset.size
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

    # Create training graphs
    test = False
    train_model = functools.partial(
        _setup_model, network=net_func, n_classes=n_classes, n_channels=n_channels, resize_size=resize_size_train, test=test)
    # Create validation graphs
    test = True
    val_model = functools.partial(
        _setup_model, network=net_func, n_classes=n_classes, n_channels=n_channels, resize_size=resize_size_val, test=test)

    # setup optimizer (SGD)
    fn = select_model_file(save_dir, start_epoch)
    nn.load_parameters(fn)
    solver.set_parameters(nn.get_parameters(grad_only=False))
    # get shuffled index using designated seed
    idx_train = get_indices(ntr, seed)
    idx_val = get_indices(nval, seed)
    idx_test = get_indices(ntest, seed)

    # training
    info = np.load(info_file_path, allow_pickle=True)
    score = []
    loss_train = None
    for epoch in tqdm(range(start_epoch, end_epoch), desc='retrain'):
        for step_info in info[epoch]:
            test = False
            idx, seeds, lr = step_info['idx'], step_info[
                'seeds'], step_info['lr']
            X, y = get_batch_data(trainset, idx_train, idx, resize_size_train,
                                  test=test, seeds=seeds, escape_list=escape_list)
            if len(X) == 0:
                continue
            _, loss_train, input_image_train = bsa.adjust_batch_size(
                train_model, len(X), loss_train, test=test)
            input_image_train["image"].d = X
            input_image_train["label"].d = y

            loss_train.forward()
            solver.zero_grad()
            loss_train.backward(clear_buffer=True)
            for key, param in nn.get_parameters(grad_only=False).items():
                param.g *= len(X) / idx.size
            solver.update()
        # evaluation
        loss_val, acc_val = eval_model(
            val_model, bsa, solver, valset, idx_val, batch_size, resize_size_val)
        loss_test, acc_test = eval_model(
            val_model, bsa, solver, testset, idx_test, batch_size, resize_size_val)
        score.append((loss_val, loss_test, acc_val, acc_test))
        # save
    save_to_csv(filename=score_filename, header=[
                'val_loss', 'test_loss', 'val_accuracy', 'test_accuracy', ], list_to_save=score, data_type='float,float,float,float')


def get_escape_list(infl_filename, n_to_remove):
    infl_list = read_csv(infl_filename)
    infl_arr = np.array(infl_list)
    header = infl_list[0]
    idx_ds = header.index('datasource_index')
    escape_list = infl_arr[1:1 + n_to_remove, idx_ds].astype(int).tolist()
    return escape_list


def run_train_for_eval(model_info_dict, file_dir_dict, n_to_remove):
    os.environ['NNABLA_CUDNN_DETERMINISTIC'] = '1'
    os.environ['NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC'] = '1'

    # files and dirs
    infl_filename = file_dir_dict['infl_filename']
    # gpu/cpu
    ctx = get_context(model_info_dict['device_id'])
    nn.set_default_context(ctx)

    # train
    escape_list = get_escape_list(infl_filename, n_to_remove)
    retrain(model_info_dict, file_dir_dict, escape_list=escape_list)
