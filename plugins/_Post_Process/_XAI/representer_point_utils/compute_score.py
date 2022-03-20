# Copyright (c) 2022 Sony Group Corporation. All Rights Reserved.
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

import argparse
import os
import random
from distutils.util import strtobool
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from re import L
from stat import FILE_ATTRIBUTE_OFFLINE
=======
>>>>>>> fe1dfc3 (first commit)
=======
from re import L
from stat import FILE_ATTRIBUTE_OFFLINE
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
>>>>>>> 3622db8 (fix output files)


import csv
import h5py
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
from nnabla.ext_utils import get_extension_context
<<<<<<< HEAD
<<<<<<< HEAD
from scipy.stats.stats import pearsonr
<<<<<<< HEAD
import nnabla.communicators as C
<<<<<<< HEAD
=======
from cifar10_load import data_source_cifar10
>>>>>>> fe1dfc3 (first commit)
=======
from scipy.stats.stats import pearsonr
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
>>>>>>> a1273c6 (first commit)
=======
import matplotlib.pyplot as plt
>>>>>>> 1322f62 (visualize correlation of output)


def label_shuffle(labels_, shuffle_rate, seed):
    random.seed(seed)
    num_cls = int(np.max(labels_)) + 1
    raw_label = labels_.copy()
    extract_num = int(len(labels_) * shuffle_rate // 10)
    for i in range(num_cls):
        extract_ind = np.where(raw_label == i)[0]
        labels = [j for j in range(num_cls)]
        labels.remove(i)  # candidate of shuffle label
        artificial_label = [
            labels[int(i) % (num_cls - 1)] for i in range(int(extract_num))
        ]
        artificial_label = np.array(
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
            random.sample(artificial_label,
                          len(artificial_label))).astype("float32")
=======
            random.sample(artificial_label, len(artificial_label))
        ).astype("float32")
>>>>>>> fe1dfc3 (first commit)
=======
            random.sample(artificial_label,
                          len(artificial_label))).astype("float32")
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
            random.sample(artificial_label, len(artificial_label))
        ).astype("float32")
>>>>>>> 1322f62 (visualize correlation of output)
        convert_label = np.array([i for _ in range(len(extract_ind))])
        convert_label[-extract_num:] = artificial_label

        labels_[extract_ind] = convert_label.reshape(-1, 1)

    return raw_label, labels_


<<<<<<< HEAD
<<<<<<< HEAD
def loss_function(args, pred, y_output, N):
    # calculate cross entropy loss

    log_softmax = F.log_softmax(pred, axis=1)
    Phi = F.sum(-y_output * log_softmax)
    # calculate l2 norm of affine layer
    l2 = 0
    for param in nn.get_parameters().values():
        l2 += F.sum(param**2)
    loss = l2 * args.lmbd + Phi / N

    return loss, Phi, l2, log_softmax


<<<<<<< HEAD
def backtracking_line_search(grad_norm,
                             x,
                             y,
                             loss,
                             N,
                             val,
                             l2,
                             threshold=1e-10):
=======
def load_cifar_labels(shuffle=False):

    train_data_source = data_source_cifar10(
        train=True, shuffle=True, label_shuffle=shuffle
    )
    test_data_source = data_source_cifar10(
        train=False, shuffle=False, label_shuffle=False
    )
    y_train = train_data_source.labels
    y_test = test_data_source.labels

    if shuffle:
        y_raw, y_train = label_shuffle(y_train, shuffle_rate=0.1, seed=0)

    return y_train, y_test


def loss_function(pred, y_output, N):
=======
def loss_function(args, pred, y_output, N):
>>>>>>> c5bf7bc (fix to pretrained nnc)
    # calculate cross entropy loss

    log_softmax = F.log_softmax(pred, axis=1)
    Phi = F.sum(-y_output * log_softmax)
    # calculate l2 norm of affine layer
    l2 = 0
    for param in nn.get_parameters().values():
        l2 += F.sum(param**2)
    loss = l2 * args.lmbd + Phi / N

    return loss, Phi, l2, log_softmax


<<<<<<< HEAD
def backtracking_line_search(grad_norm, x, y, loss, N, val, l2, threshold=1e-10):
>>>>>>> fe1dfc3 (first commit)
=======
def backtracking_line_search(grad_norm,
                             x,
                             y,
                             loss,
                             N,
                             val,
                             l2,
                             threshold=1e-10):
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
def backtracking_line_search(grad_norm, x, y, loss, N, val, l2, threshold=1e-10):
>>>>>>> 1322f62 (visualize correlation of output)
    t = 10.0
    beta = 0.5
    params = nn.get_parameters().values()
    p_data_org_list = [F.identity(p.data) for p in params]
    p_grad_org_list = [F.identity(p.grad) for p in params]

    while True:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        for p, p_data_org, p_grad_org in zip(params, p_data_org_list,
                                             p_grad_org_list):
=======
        for p, p_data_org, p_grad_org in zip(params, p_data_org_list, p_grad_org_list):
>>>>>>> fe1dfc3 (first commit)
=======
        for p, p_data_org, p_grad_org in zip(params, p_data_org_list,
                                             p_grad_org_list):
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
        for p, p_data_org, p_grad_org in zip(params, p_data_org_list, p_grad_org_list):
>>>>>>> 1322f62 (visualize correlation of output)
            p.data.copy_from(p_data_org - t * p_grad_org)

        loss.forward()
        if t < threshold:
            print("t too small")
            break
<<<<<<< HEAD
<<<<<<< HEAD
        if (loss.d - val + t * grad_norm.data**2 / 2) >= 0:
=======
        if (loss.d - val + t * grad_norm.data ** 2 / 2) >= 0:
>>>>>>> fe1dfc3 (first commit)
=======
        if (loss.d - val + t * grad_norm.data**2 / 2) >= 0:
>>>>>>> c5bf7bc (fix to pretrained nnc)
            t = beta * t
        else:
            break

    return params


<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
def calculate_alpha(args, parameter_list, X, Y, Y_label, feature_valid, solver,
                    output_valid, pred, loss, phi, l2, lg):
    print('## Now Score Calculating')
    min_loss = 10000.0
    feature_valid.d = X
    output_valid.d = Y

=======
def calculate_alpha(
=======
def calculate_alpha(
    args,
>>>>>>> 1322f62 (visualize correlation of output)
    parameter_list,
    X,
    Y,
    Y_label,
    feature_valid,
    solver,
    output_valid,
    softmax_valid,
    pred,
    loss,
    phi,
    l2,
<<<<<<< HEAD
):

    min_loss = 10000.0
    feature_valid.d = X
    output_valid.d = Y
>>>>>>> fe1dfc3 (first commit)
=======
def calculate_alpha(args, parameter_list, X, Y, Y_label, feature_valid, solver,
                    output_valid, pred, loss, phi, l2, lg):
    print('## Now Score Calculating')
=======
    lg,
):
    print("## Now Score Calculating")
>>>>>>> 1322f62 (visualize correlation of output)
    min_loss = 10000.0
    feature_valid.d = X
    output_valid.d = Y

>>>>>>> c5bf7bc (fix to pretrained nnc)
    for epoch in range(args.epoch):
        phi_loss = 0

        loss.forward()
        solver.zero_grad()
        loss.backward()
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

        phi_loss = phi.d / len(X)
        temp_W = parameter_list

        grad_loss = F.add_n(
<<<<<<< HEAD
            *[F.mean(F.abs(p.grad)) for p in nn.get_parameters().values()])
        grad_norm = F.add_n(
            *[F.norm(p.grad) for p in nn.get_parameters().values()])
=======
        phi_loss = phi.d / len(X)
        temp_W = parameter_list
        grad_loss = F.add_n(
            *[F.mean(F.abs(p.grad)) for p in nn.get_parameters().values()]
        )
        grad_norm = F.add_n(*[F.norm(p.grad)
                            for p in nn.get_parameters().values()])
>>>>>>> fe1dfc3 (first commit)
=======
        print(loss.d)
        print(l2.d)
        print(phi.d)
=======
>>>>>>> a1273c6 (first commit)

        phi_loss = phi.d / len(X)
        temp_W = parameter_list

        grad_loss = F.add_n(
            *[F.mean(F.abs(p.grad)) for p in nn.get_parameters().values()])
        grad_norm = F.add_n(
            *[F.norm(p.grad) for p in nn.get_parameters().values()])
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
            *[F.mean(F.abs(p.grad)) for p in nn.get_parameters().values()]
        )
<<<<<<< HEAD
        grad_norm = F.add_n(*[F.norm(p.grad) for p in nn.get_parameters().values()])
>>>>>>> 1322f62 (visualize correlation of output)
=======
        grad_norm = F.add_n(*[F.norm(p.grad)
                            for p in nn.get_parameters().values()])
>>>>>>> 3622db8 (fix output files)

        if grad_loss.data < min_loss:
            if epoch == 0:
                init_grad = grad_loss.data
            min_loss = grad_loss.data
            best_W = temp_W
            if min_loss < init_grad / 200:
                print("stopping criteria reached in epoch :{}".format(epoch))
                break
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        parameter_list = backtracking_line_search(grad_norm, X, Y, loss,
                                                  len(X), loss.d, l2)
        if epoch % 100 == 0:
            print("Epoch:{:4d}\tloss:{}\tphi_loss:{}\tl2(lmbd):{}\tgrad:{}".
                  format(epoch, loss.d, phi_loss, args.lmbd * l2.d,
                         grad_loss.data))
=======
        parameter_list = backtracking_line_search(
            grad_norm, X, Y, loss, len(X), loss.d, l2
        )
        if epoch % 100 == 0:
=======
        parameter_list = backtracking_line_search(
            grad_norm, X, softmax_valid.d, loss, len(X), loss.d, l2
        )
        if epoch % 100 == 0:
>>>>>>> 1322f62 (visualize correlation of output)
            print(
                "Epoch:{:4d}\tloss:{}\tphi_loss:{}\tl2(lmbd):{}\tgrad:{}".format(
                    epoch, loss.d, phi_loss, args.lmbd * l2.d, grad_loss.data
                )
            )
<<<<<<< HEAD
>>>>>>> fe1dfc3 (first commit)
=======
        parameter_list = backtracking_line_search(grad_norm, X, Y, loss,
                                                  len(X), loss.d, l2)
        if epoch % 100 == 0:
            print("Epoch:{:4d}\tloss:{}\tphi_loss:{}\tl2(lmbd):{}\tgrad:{}".
                  format(epoch, loss.d, phi_loss, args.lmbd * l2.d,
                         grad_loss.data))
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
>>>>>>> 1322f62 (visualize correlation of output)

    for weight, param in zip(nn.get_parameters().values(), best_W):
        weight.data.copy_from(param.data)

    softmax_value = F.softmax(pred)
    softmax_value.forward()
    # derivative of softmax cross entropy
    weight_matrix = softmax_value.d - softmax_valid.d
    weight_matrix = np.divide(weight_matrix, (-2.0 * args.lmbd * len(Y)))
<<<<<<< HEAD

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    np.save(os.path.join(args.monitor_path, "weight_matrix.npy"),
            weight_matrix)
=======
    np.save(os.path.join(data_dir, "weight_matrix.npy"), weight_matrix)
>>>>>>> fe1dfc3 (first commit)
=======
    np.save(os.path.join(args.output, "weight_matrix.npy"), weight_matrix)
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
    np.save(os.path.join(args.monitor_path, "weight_matrix.npy"),
            weight_matrix)
>>>>>>> a1273c6 (first commit)
=======
=======
>>>>>>> a73a4ea (bug fix)
    np.save(os.path.join(args.monitor_path, "weight_matrix.npy"), weight_matrix)
>>>>>>> 1322f62 (visualize correlation of output)

    # computer alpha
    alpha = []
    print(weight_matrix.shape)
    for ind, label in enumerate(Y_label.reshape(-1)):
        alpha.append(float(weight_matrix[ind, int(label)]))
    alpha = np.abs(np.array(alpha))
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    np.save(os.path.join(args.monitor_path, "alpha_vgg_nnabla_score.npy"),
            alpha)
=======
    np.save(os.path.join(data_dir, "alpha_vgg_nnabla_score.npy"), alpha)
>>>>>>> fe1dfc3 (first commit)
=======
    np.save(os.path.join(args.output, "alpha_vgg_nnabla_score.npy"), alpha)
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
    np.save(os.path.join(args.monitor_path, "alpha_vgg_nnabla_score.npy"),
            alpha)
>>>>>>> a1273c6 (first commit)
=======
    np.save(os.path.join(args.monitor_path, "alpha_vgg_nnabla_score.npy"), alpha)
>>>>>>> 1322f62 (visualize correlation of output)

    # calculate correlation
    w = np.matmul(X.T, weight_matrix)
    temp = np.matmul(X, w)
    softmax_value = F.softmax(nn.Variable.from_numpy_array(temp))
    softmax_value.forward()
    y_p = softmax_value.d

    print(
        "L1 difference between ground truth prediction and prediction by representer theorem decomposition"
    )
    print(np.mean(np.abs(softmax_valid.d - y_p)))

<<<<<<< HEAD
<<<<<<< HEAD
=======
    from scipy.stats.stats import pearsonr

>>>>>>> fe1dfc3 (first commit)
=======
>>>>>>> c5bf7bc (fix to pretrained nnc)
    print(
        "pearson correlation between ground truth  prediction and prediciton by representer theorem"
    )
    corr, _ = pearsonr(Y.reshape(-1), y_p.reshape(-1))
    print(corr)

    return weight_matrix


<<<<<<< HEAD
<<<<<<< HEAD
def compute_score(args, info):

    ctx = get_extension_context(
        ext_name="cudnn", device_id=args.device_id, type_config=args.type_config
    )

    nn.set_default_context(ctx)

    with h5py.File(info, "r") as hf:
        train_feature = hf["feature"]["train"][:]
        test_feature = hf["feature"]["test"][:]
        train_output = hf["output"]["train"][:]
        test_output = hf["output"]["test"][:]
        Y_label = hf["label"]["train"][:][: len(train_feature)]
        parameter_list = [hf["param"]["weight"][:], hf["param"]["bias"][:]]

    feature_dict = {"train": train_feature, "test": test_feature}
    output_dict = {"train": train_output, "test": test_output}
    output_dict_softmax = dict()
    for phase, pre_act in output_dict.items():
        softmax_ = F.softmax(nn.Variable.from_numpy_array(pre_act))
        softmax_.forward()
        output_dict_softmax[phase] = softmax_.d
    feature_shape = train_feature.shape
    n_cls = train_output.shape[-1]

    nn.clear_parameters()
    # computation graph
    feature_valid = nn.Variable((feature_shape[0], feature_shape[1]))
    output_valid = nn.Variable((feature_shape[0], n_cls))
    softmax_valid = F.softmax(output_valid)

    with nn.parameter_scope("final_layer"):
        pred = PF.affine(feature_valid, n_cls)

<<<<<<< HEAD
<<<<<<< HEAD
    loss, phi, l2, lg = loss_function(args, pred, output_valid, len(X_feature))
=======
def main():
=======
def compute_score(args, info):

<<<<<<< HEAD
>>>>>>> c5bf7bc (fix to pretrained nnc)
    extension_module = args.context
    ctx = get_extension_context(extension_module,
=======
    ctx = get_extension_context(ext_name="cudnn",
>>>>>>> a1273c6 (first commit)
                                device_id=args.device_id,
                                type_config=args.type_config)

    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = mpi_rank
    ctx = get_extension_context('cudnn', device_id=device_id)

    nn.set_default_context(ctx)

    try:
        with h5py.File(info, "r") as hf:
            X_feature = hf["feature"]["train"][:]
            Y_train = hf["output"]["train"][:]
            Y_label = hf["label"][:][:len(X_feature)]
            parameter_list = [hf["param"]["weight"][:], hf["param"]["bias"][:]]

    except:
        parameter_list = [info['param'][0], info['param'][1]]  ## weight, bias
        X_feature = info['feature']['train'][:]
        Y_train = info["output"]['train'][:]
        Y_label = info['label'][:][:len(X_feature)]

    feature_shape = X_feature.shape
    n_cls = Y_train.shape[-1]

    nn.clear_parameters()
    # computation graph
    feature_valid = nn.Variable((feature_shape[0], feature_shape[1]))
    output_valid = nn.Variable((feature_shape[0], n_cls))

    with nn.parameter_scope("final_layer"):
<<<<<<< HEAD
        pred = PF.affine(feature_valid, 10)
    loss, phi, l2 = loss_function(pred, output_valid, len(X_feature))
>>>>>>> fe1dfc3 (first commit)
=======
        pred = PF.affine(feature_valid, n_cls)

    loss, phi, l2, lg = loss_function(args, pred, output_valid, len(X_feature))
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
    loss, phi, l2, lg = loss_function(args, pred, output_valid, len(train_feature))
>>>>>>> 1322f62 (visualize correlation of output)
=======
    loss, phi, l2, lg = loss_function(
<<<<<<< HEAD
        args, pred, output_valid, len(train_feature))
>>>>>>> 3622db8 (fix output files)
=======
        args, pred, softmax_valid, len(train_feature))
>>>>>>> 381b077 (fix network architecture bug)

    # parameter initialized
    for weight, param in zip(nn.get_parameters().values(), parameter_list):
        weight.data.copy_from(nn.NdArray.from_numpy_array(param))

    solver = S.Sgd(lr=1.0)
    solver.set_parameters(nn.get_parameters())
    for ind, (name, param) in enumerate(nn.get_parameters().items()):
        param.grad.zero()
        param.need_grad = True

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    calculate_alpha(args, parameter_list, X_feature, Y_train, Y_label,
                    feature_valid, solver, output_valid, pred, loss, phi, l2,
                    lg)
=======
    calculate_alpha(
        parameter_list,
        X_feature,
        Y_train,
=======
    weight_matrix = calculate_alpha(
        args,
        parameter_list,
        train_feature,
        train_output,
>>>>>>> 1322f62 (visualize correlation of output)
        Y_label,
        feature_valid,
        solver,
        output_valid,
        softmax_valid,
        pred,
        loss,
        phi,
        l2,
<<<<<<< HEAD
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmbd", type=float, default=0.003)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--shuffle_label", type=strtobool)
    parser.add_argument(
        "--context",
        "-c",
        type=str,
        default="cudnn",
        help="Extension path. ex) cpu, cudnn.",
    )
    parser.add_argument(
        "--device_id",
        "-d",
        type=str,
        default="0",
        help="Device ID the training run on. This is only valid if you specify `-c cudnn`.",
    )
    parser.add_argument(
        "--type_config",
        "-t",
        type=str,
        default="float",
        help='Type of computation. e.g. "float", "half".',
    )

    args = parser.parse_args()
    input_dir = (
        "./data/input/shuffle" if args.shuffle_label else "./data/input/no_shuffle"
    )
    data_dir = "./data/info/shuffle" if args.shuffle_label else "./data/info/no_shuffle"

    print(args)
    main()
>>>>>>> fe1dfc3 (first commit)
=======
    calculate_alpha(args, parameter_list, X_feature, Y_train, Y_label,
                    feature_valid, solver, output_valid, pred, loss, phi, l2,
                    lg)
>>>>>>> c5bf7bc (fix to pretrained nnc)
=======
        lg,
    )

    predicted_dict = {}

    for phase, feature in feature_dict.items():
        w = np.matmul(train_feature.T, weight_matrix)
        temp = np.matmul(feature, w)
        softmax = F.softmax(nn.Variable.from_numpy_array(temp))
        softmax.forward()
        predicted_dict[phase] = softmax.d

    np.random.seed(401)
    train_rand_idx = np.random.choice(50000, 2000, replace=False)
    np.random.seed(4)
    test_rand_idx = np.random.choice(10000, 1000, replace=False)
    rand_idx_dict = {"train": train_rand_idx, "test": test_rand_idx}

    for ind, phase in enumerate(["train", "test"]):

        info_list = [["actual_output", "approximated_output"]]
        for ind, (act, pred) in enumerate(
            zip(output_dict_softmax[phase], predicted_dict[phase])
        ):

            if ind in rand_idx_dict[phase]:
                info_list.extend(
                    [[float(act_), float(pred_)]
                     for act_, pred_ in zip(act, pred)]
                )
        if os.path.exists(f"{phase}_output_correlation.csv"):
            os.remove(f"{phase}_output_correlation.csv")

<<<<<<< HEAD
    plt.tight_layout()
    plt.savefig("correlation.png", dpi=200)
>>>>>>> 1322f62 (visualize correlation of output)
=======
        with open(f"{phase}_output_correlation.csv", "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(info_list)
>>>>>>> 3622db8 (fix output files)
