# Copyright 2022 Sony Group Corporation.
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
from __future__ import absolute_import
import os
import functools
import math
import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
import nnabla.utils.save as save
from tqdm import tqdm
from nnabla.utils.data_iterator import data_iterator
from .args import get_train_args
from .datasource import get_datasource
from .model import resnet56_prediction, loss_function, resnet23_prediction
from .utils import get_context, ensure_dir
from .utils import read_yaml, create_learning_rate_scheduler
from .utils import save_nnp, save_checkpoint


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def eval(val_iterator, image_valid, label_valid, loss_val, pred_valid, bs_valid):
    # Validation
    n_val_samples = val_iterator._data_source.size
    val_iter = math.ceil(n_val_samples / bs_valid)
    ve = 0.0
    vloss = 0.0
    for j in range(val_iter):
        image, label, *_ = val_iterator.next()
        image_valid.d = image
        label_valid.d = label
        loss_val.forward()
        vloss += loss_val.data.data.copy() * bs_valid
        ve += categorical_error(pred_valid.d, label)
    ve /= val_iter
    vloss /= n_val_samples


def train(args, need_eval=False):
    bs_train, bs_valid = args.train_batch_size, args.val_batch_size
    ctx = get_context(device_id=args.device_id)
    nn.set_default_context(ctx)

    if args.weight_input:
        train_data_source = get_datasource(
            filename=args.input_train, shuffle=True, label_shuffle=args.shuffle_label)
        n_train_samples = train_data_source.size
        # Data Iterator
        train_iterator = data_iterator(
            train_data_source, bs_train, None, False, False)
        ensure_dir(args.weight_output)
        train_data_source.save_to_csv(os.path.join(
            args.weight_output, 'data_train.csv'))

    if args.model == "resnet23":
        model_prediction = resnet23_prediction
    elif args.model == "resnet56":
        model_prediction = resnet56_prediction
    ncls = train_data_source.get_n_classes()
    image, *_ = train_data_source._get_data(0)
    image_shape = image.shape
    prediction = functools.partial(
        model_prediction, ncls=ncls, nmaps=64, act=F.relu, seed=args.seed)
    # Create training graphs
    test = False
    image_train = nn.Variable(
        (bs_train, image_shape[0], image_shape[1], image_shape[2]))
    label_train = nn.Variable((bs_train, 1))
    pred_train, _ = prediction(image_train, test)

    loss_train = loss_function(pred_train, label_train)

    if need_eval:
        val_data_source = get_datasource(filename=args.input_val)
        val_iterator = data_iterator(
            val_data_source, bs_valid, None, False, False)
        image, *_ = val_data_source._get_data(0)
        image_shape = image.shape
        # Create validation graph
        test = True
        image_valid = nn.Variable(
            (bs_valid, image_shape[0], image_shape[1], image_shape[2]))
        label_valid = nn.Variable((bs_valid, 1))
        pred_valid, _ = prediction(image_valid, test)
        loss_val = loss_function(pred_valid, label_valid)
        # save_nnp
        contents = save_nnp({"x": image_valid}, {"y": pred_valid}, bs_valid)
        save.save(
            os.path.join(args.model_save_path,
                         (args.model + "_epoch0_result.nnp")), contents
        )

    for param in nn.get_parameters().values():
        param.grad.zero()
    dir_path = os.path.abspath(os.path.dirname(__file__))
    cfg = read_yaml(os.path.join(dir_path, "learning_rate.yaml"))
    lr_sched = create_learning_rate_scheduler(cfg.learning_rate_config)

    solver = S.Momentum(momentum=0.9, lr=lr_sched.get_lr())
    solver.set_parameters(nn.get_parameters())
    start_point = 0

    train_iter = math.ceil(n_train_samples / bs_train)
    # Training-loop
    for i in tqdm(range(start_point, args.train_epochs)):
        lr_sched.set_epoch(i)
        solver.set_learning_rate(lr_sched.get_lr())
        if need_eval:
            eval(val_iterator, image_valid, label_valid,
                 loss_val, pred_valid, bs_valid)
        if int(i % args.model_save_interval) == 0:
            # save checkpoint file
            save_checkpoint(args.model_save_path, i, solver)

        # Forward/Zerograd/Backward
        e = 0.0
        loss = 0.0
        for k in range(train_iter):
            image, label, *_ = train_iterator.next()
            image_train.d = image
            label_train.d = label
            loss_train.forward()
            solver.zero_grad()
            loss_train.backward()
            solver.update()
            e += categorical_error(pred_train.d, label_train.d)
            loss += loss_train.data.data.copy() * bs_train
        e /= train_iter
        loss /= n_train_samples

        e = categorical_error(pred_train.d, label_train.d)

    nn.save_parameters(
        os.path.join(args.model_save_path, "params_%06d.h5" %
                     (args.train_epochs))
    )

    if need_eval:
        # save_nnp_lastepoch
        contents = save_nnp({"x": image_valid}, {"y": pred_valid}, bs_valid)
        save.save(os.path.join(args.model_save_path,
                               (args.model + "_result.nnp")), contents)


if __name__ == "__main__":
    args = get_train_args()
    train(args)
