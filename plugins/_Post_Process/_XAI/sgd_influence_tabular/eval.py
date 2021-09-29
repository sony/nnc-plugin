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
from tqdm import tqdm
import numpy as np
import nnabla as nn
import nnabla.functions as F
from sgd_influence_tabular.network import get_solver, select_model, calc_acc, get_n_classes
from sgd_influence_tabular.network import get_batch_data, get_indices, setup_dataset
from utils.file import get_context, read_csv, save_to_csv


def select_model_file(output_path, start_epoch):
    fn = '%s/epoch%02d/weights/initial_model.h5' % (output_path, start_epoch)
    return fn


def eval_model(bs_adjuster, val_model, dataset, idx_list_to_data, batch_size, alpha):
    loss = 0
    acc = 0
    n = len(idx_list_to_data)
    idx = np.array_split(np.arange(n), batch_size)
    loss_fn = None
    for _, i in enumerate(idx):
        X, y = get_batch_data(dataset, idx_list_to_data, i)
        if len(i) == 0:
            continue
        pred, loss_fn, input_data = bs_adjuster.adjust_batch_size(
            val_model, len(X), test=True)
        input_data["feat"].d = X
        input_data["label"].d = y
        for _k, p in nn.get_parameters(grad_only=False).items():
            loss_fn += 0.5 * alpha * F.sum(p * p)
        loss_fn.forward()
        loss += loss_fn.d * len(X)
        acc += calc_acc(pred.d, y, method='sum')
    loss /= n
    acc /= n
    return loss, acc


def retrain(config, escape_list=[]):
    # params
    n_classes = get_n_classes(config.train_csv, config.label_variable)
    lr = config.lr
    bsa = config.bs_adjuster
    seed = config.seed
    batch_size = config.batch_size
    eval_end_epoch = config.num_epochs
    eval_start_epoch = config.eval_start_epoch
    # files and dirs
    weight_output = config.weight_output
    info_filename = config.info_filename
    info_file_path = os.path.join(weight_output, info_filename)
    # setup
    trainset, n_features = setup_dataset(config.train_csv)
    valset, _ = setup_dataset(config.val_csv)
    testset, _ = setup_dataset(config.test_csv)
    n_tr = trainset.size
    n_val = valset.size
    n_test = testset.size
    network = config.network
    _setup_model = functools.partial(select_model, network=network, n_features=n_features,
                                     net_name_dict=config.net_name_dict, n_classes=n_classes)

    # Create training graphs
    train_model = functools.partial(_setup_model, test=False)
    # Create validation graphs
    val_model = functools.partial(_setup_model, test=True)
    solver = get_solver(network, config.network_info, lr)
    # get shuffled index using designated seed
    idx_train = get_indices(n_tr, seed)
    idx_val = get_indices(n_val, seed)
    idx_test = get_indices(n_test, seed)

    # training
    info = np.load(info_file_path, allow_pickle=True)
    score = []
    loss_train = None
    for epoch in tqdm(range(eval_start_epoch, eval_end_epoch), desc='retrain'):
        # set params
        fn = select_model_file(weight_output, epoch)
        nn.load_parameters(fn)
        solver.set_parameters(nn.get_parameters(grad_only=False))
        for step_info in info[epoch]:
            idx, lr, alpha = step_info['idx'], step_info['lr'], step_info['alpha']
            X, y = get_batch_data(trainset, idx_train,
                                  idx, escape_list=escape_list)
            params = nn.get_parameters(grad_only=False)
            if len(X) == 0:
                continue
            _, loss_train, input_data_train = bsa.adjust_batch_size(
                train_model, len(X), loss_train)
            solver.set_parameters(params)
            solver.set_learning_rate(lr)
            input_data_train["feat"].d = X
            input_data_train["label"].d = y
            for _k, p in nn.get_parameters(grad_only=False).items():
                loss_train += 0.5 * alpha * F.sum(p * p)
            loss_train.forward()
            solver.zero_grad()
            loss_train.backward()
            for key, param in nn.get_parameters(grad_only=False).items():
                param.g *= len(X) / idx.size
            solver.update()
        # evaluation
        loss_val, acc_val = eval_model(
            bsa, val_model, valset, idx_val, batch_size, alpha)
        loss_test, acc_test = eval_model(
            bsa, val_model, testset, idx_test, batch_size, alpha)
        score.append((loss_val, loss_test, acc_val, acc_test))
        # save
    save_to_csv(filename=config.score_output, header=[
                'val_loss', 'test_loss', 'val_accuracy', 'test_accuracy', ], list_to_save=score, data_type='float,float,float,float')


def get_escape_list(infl_filename, n_to_remove):
    infl_list = read_csv(infl_filename)
    infl_arr = np.array(infl_list)
    header = infl_list[0]
    idx_ds = header.index('index')
    escape_list = infl_arr[1:1+n_to_remove, idx_ds].astype(int).tolist()
    return escape_list


def run_train_for_eval(config, n_to_remove):
    os.environ['NNABLA_CUDNN_DETERMINISTIC'] = '1'
    os.environ['NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC'] = '1'
    # gpu/cpu
    ctx = get_context(config.device_id)
    nn.set_default_context(ctx)

    # train
    escape_list = get_escape_list(config.infl_filepath, n_to_remove)
    retrain(config, escape_list=escape_list)
