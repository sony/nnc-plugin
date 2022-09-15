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
import os
import sys
import shutil
import argparse
import zipfile
import google.protobuf.text_format as text_format
import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
from nnabla import logger
import nnabla.parametric_functions as PF
from nnabla.utils import nnabla_pb2
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from nnabla.utils.save import save


def classifier(features_n, n_hidden=32):
    """
    Classifier network
    """
    with nn.parameter_scope('classifier'):
        l1 = PF.affine(features_n, n_hidden, name='l1')
        l1 = F.relu(l1)
        l2 = PF.affine(l1, n_hidden, name='l2')
        l2 = F.relu(l2)
        l3 = PF.affine(l2, n_hidden, name='l3')
        l3 = F.relu(l3)
        l4 = PF.affine(l3, 1, name='l4')
    return l4


def adversary(clf_out, n_hidden=32):
    """
    Adversarial network
    """
    with nn.parameter_scope("adversary"):
        Al1 = PF.affine(clf_out, n_hidden, name='Al1')
        Al1 = F.relu(Al1)
        Al2 = PF.affine(Al1, n_hidden, name='Al2')
        Al2 = F.relu(Al2)
        Al3 = PF.affine(Al2, n_hidden, name='Al3')
        Al3 = F.relu(Al3)
        Al4 = PF.affine(Al3, 1, name='Al4')
    return Al4


def get_args():
    """
    parse args
    """
    parser = argparse.ArgumentParser(
        description='Adversarial Debiasing\n' +
                    '\n' +
                    'Adversarial Debiasing\n' +
                    'Specify both Training and Validation datasets in the DATASET tab\n' +
                    'Specify Epoch, Batch Size in the CONFIG tab and execute this plug-in.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-c',
        '--config',
        help='config file(nntxt) default=net.nntxt',
        required=True,
        default='net.nntxt')
    parser.add_argument(
        '-o',
        '--output_dir',
        help='output_dir(dir)',
        required=True)
    parser.add_argument(
        '-m',
        '--mode',
        help='training mode (option:classifier,adversarial_debiasing) default=classifier',
        default='classifier')
    parser.add_argument(
        '-lamda',
        '--lamda',
        help='Specify Adversarial loss parameter',
        type=float)
    parser.add_argument(
        '-adv_epoch',
        '--adv_epoch',
        help='Specify the no of epochs to train Adversarial network default=100',
        type=int)
    parser.add_argument(
        '-zp', '--privileged_variable',
        help='specify the privileged variable from the input CSV file (variable)',
        required=True)
    parser.add_argument(
        '-zu', '--unprivileged_variable',
        help='specify the unprivileged variable from the input CSV file (variable)',
        required=True)
    return parser


def func(args):
    # Open config
    logger.log(99, 'Loading config...')
    proto = nnabla_pb2.NNablaProtoBuf()
    with open(args.config, mode='r') as f:
        text_format.Merge(f.read(), proto)

    batch_size = proto.dataset[0].batch_size
    clf_epoch = proto.training_config.max_epoch
    iter_epoch = proto.training_config.iter_per_epoch
    monitor_interval = proto.training_config.monitor_interval
    # data loader
    data_iterator_train = (lambda: data_iterator_csv_dataset(
        uri=proto.dataset[0].uri,
        batch_size=batch_size,
        shuffle=True,
        normalize=False,
        with_memory_cache=False,
        with_file_cache=False))

    try:
        di = data_iterator_train()
        vind_x = di.variables.index('x')
        vind_y = di.variables.index('y')
        vind_z = di.variables.index('z')
        data_x = di.next()[vind_x]
    except:
        logger.critical(
            f'Variable "x", "y" and/or "z" is not found in the training dataset')
        raise
    # input variables for the network
    n_feature = nn.Variable((batch_size, data_x.shape[1]))
    n_label = nn.Variable((batch_size, 1))
    n_senstive = nn.Variable((batch_size, 1))
    test_feature = nn.Variable((1, data_x.shape[1]))

    # classifier training graph
    clf = classifier(n_feature)
    clf.persistent = True
    loss = F.mean(F.sigmoid_cross_entropy(clf, n_label))
    loss.persistent = True
    # classifier validation graph
    v_clf = classifier(test_feature)
    v_clf_out = F.sigmoid(v_clf)
    learning_rate = 1e-03
    clf_solver = S.Adam(learning_rate)
    logger.log(
        99, f"Maximum number of epochs to train the Classifier network : {clf_epoch}")
    logger.log(
        99, f"Adam solver is used to train the classifier, learning rate {learning_rate}", )
    with nn.parameter_scope("classifier"):
        clf_solver.set_parameters(nn.get_parameters())
    di = data_iterator_train()
    # Train the classifier network first in both the
    # "classifier" and "adversarial_debiasing" training modes
    if args.mode == "classifier" or args.mode == "adversarial_debiasing":
        logger.log(99, "Classifier Training Started")
        for epoch in range(clf_epoch):
            loss_list = []
            for i in range(iter_epoch):
                data = di.next()
                n_feature.d = data[vind_x].reshape(n_feature.shape)
                n_label.d = data[vind_y].reshape(n_label.shape)
                clf_solver.zero_grad()
                loss.forward(clear_no_need_grad=True)
                loss.backward(clear_buffer=True)
                clf_solver.update()
                loss_list.append(float(loss.d))

            if epoch % monitor_interval == 0:
                logger.log(
                    99, f"Epoch : {epoch} , Classifier Loss: {sum(loss_list) / len(loss_list)}")
                if args.mode == "classifier":
                    f = open(os.path.join(
                        args.output_dir, 'monitoring_report.yml'), 'a')
                    f.write('{}:\n'.format(epoch))
                    f.write('  cost: {}\n'.format(
                        sum(loss_list) / len(loss_list)))
                    f.close()
    else:
        logger.critical(f'{args.mode} is not supported.')
        sys.exit(1)
    # train the adversarial network
    if args.mode == "adversarial_debiasing":
        logger.log(99, "Adversarial Training Started")
        privileged_index = args.privileged_variable.split(":")[
            0].split("__")[1]
        unprivileged_index = args.unprivileged_variable.split(":")[
            0].split("__")[1]
        lamdas = args.lamda
        adv_epoch = args.adv_epoch
        # Adversarial training graph
        adv = adversary(clf)
        adv_loss = F.mean(F.mul_scalar(
            F.sigmoid_cross_entropy(adv, n_senstive), lamdas))
        adv_solver = S.Adam(learning_rate)
        logger.log(
            99, f"Maximum number of epochs to train the Adversary network : {adv_epoch}")
        logger.log(99, f"Adam solver is used to train the Adversary network, "
                       f"learning rate of {learning_rate}", )
        with nn.parameter_scope("adversary"):
            adv_solver.set_parameters(nn.get_parameters())
        # loss
        clfloss = loss - adv_loss
        # train the adversary network
        for epoch in range(adv_epoch):
            adv_loss_list = []
            clf.need_grad = False
            for i in range(iter_epoch):
                data_adv = di.next()
                n_feature.d = data_adv[vind_x].reshape(n_feature.shape)
                n_label.d = data_adv[vind_y].reshape(n_label.shape)
                privileged_variables = data_adv[vind_z][:, int(
                    privileged_index)] == 1
                unprivileged_variables = data_adv[vind_z][:, int(
                    unprivileged_index)] == 0
                if not (privileged_variables == unprivileged_variables).all():
                    logger.log(99, f"Both privileged and unprivileged "
                                   f"variable values should not be same")
                    sys.exit(0)
                n_senstive.d = data_adv[vind_z][:, int(
                    privileged_index)].reshape(n_senstive.shape)
                adv_solver.zero_grad()
                adv_loss.forward()
                adv_loss.backward(clear_buffer=True)
                adv_solver.update()
                adv_loss_list.append(float(adv_loss.d))
            if epoch % monitor_interval == 0:
                logger.log(
                    99, f"Epoch : {epoch}, Adversary Loss: {sum(adv_loss_list) / len(adv_loss_list)}")
                f = open(os.path.join(
                    args.output_dir, 'monitoring_report.yml'), 'a')
                f.write('{}:\n'.format(epoch))
                f.write('  cost: {}\n'.format(
                    sum(adv_loss_list) / len(adv_loss_list)))
                f.close()

            for i in range(iter_epoch):
                data_clf = di.next()
                n_feature.d = data_clf[vind_x].reshape(n_feature.shape)
                n_label.d = data_clf[vind_y].reshape(n_label.shape)
                n_senstive.d = data_clf[vind_z][:, int(
                    privileged_index)].reshape(n_senstive.shape)
                pass
            clf.need_grad = True
            loss.forward(clear_no_need_grad=True)
            clf_solver.zero_grad()
            clfloss.forward(clear_no_need_grad=True)
            clfloss.backward(clear_buffer=True)
            clf_solver.update()
            if epoch % monitor_interval == 0:
                logger.log(99, f"Epoch {epoch}, Classifier Loss: {loss.d}")
    else:
        pass

    # Create classification model
    logger.log(99, 'Saving classification model...')
    contents = {
        'networks': [
            {'name': 'network',
             'batch_size': 1,
             'outputs': {'y\'': v_clf_out},
             'names': {'x': test_feature}}],
        'executors': [
            {'name': 'runtime',
             'network': 'network',
             'data': ['x'],
             'output': ['y\'']}]}
    # create results.nntxt
    result_config_file = os.path.join(args.output_dir, 'results.nntxt')
    save(result_config_file, contents)
    result_model_file = os.path.join(args.output_dir, 'results.nnp')
    save(result_model_file, contents)
    result_proto = nnabla_pb2.NNablaProtoBuf()
    with open(result_config_file, mode='r') as f:
        text_format.Merge(f.read(), result_proto)
    result_proto.executor[0].no_image_normalization = True
    with open(result_config_file, mode='w') as f:
        text_format.PrintMessage(result_proto, f)

    result_model_file_tmp = result_model_file + '.tmp'
    with zipfile.ZipFile(result_model_file, 'r') as zr:
        with zipfile.ZipFile(result_model_file_tmp, 'w') as zw:
            for info in zr.infolist():
                ext = os.path.splitext(info.filename)[1].lower()
                if ext == ".nntxt" or ext == ".prototxt":
                    zw.write(result_config_file, arcname=info.filename)
                else:
                    bin = zr.read(info.filename)
                    zw.writestr(info, bin)
    shutil.move(result_model_file_tmp, result_model_file)
    logger.log(99, 'Training Completed.')


def main():
    parser = get_args()
    parser.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
