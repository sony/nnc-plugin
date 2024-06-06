# Copyright 2024 Sony Group Corporation.
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
import csv
import argparse
import shutil
import subprocess
import json
import zipfile

from google.protobuf import json_format
import google.protobuf.text_format as text_format

from nnabla import logger
from nnabla.utils import nnabla_pb2
from nnabla.utils.image_utils import imread
import nnabla_nas_util.main


def NNPSolverToNNASOptimizer(args, solver):
    solver_type = solver.type
    optimizer = {
        "grad_clip": 5.0,
        "weight_decay": solver.weight_decay,
        "name": solver_type
    }
    solver_dict = json_format.MessageToDict(
        solver, preserving_proto_field_name=True)
    optimizer.update(solver_dict[solver_type.lower() + '_param'])
    if solver.lr_scheduler_type != '':
        optimizer['lr_scheduler'] = solver.lr_scheduler_type + 'Scheduler'
        logger.info(
            99, f'{optimizer["lr_scheduler"]} is used as learning rate scheduler, but the argument setting is ignored.')
    return optimizer


def func(args):
    genotype = args.arch
    progress_txt = os.path.join(args.output_dir, 'progress.txt')

    # Open config
    proto = nnabla_pb2.NNablaProtoBuf()
    with open(args.config, mode='r') as f:
        text_format.Merge(f.read(), proto)
    warmup_iter = 0
    if proto.optimizer[0].solver.lr_warmup_scheduler_type == 'Linear':
        warmup_iter = proto.optimizer[0].solver.linear_warmup_scheduler_param.warmup_iter

    # Open dataset
    with open(proto.dataset[1].uri, encoding='utf_8_sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != 2:
            logger.critical(
                f'The dataset must consist of two variables, x and y. Header of the specified data = {header}')
            sys.exit(1)
        num_class = 1
        input_shape = None
        for row in reader:
            c = int(row[1]) + 1
            if c > num_class:
                num_class = c
            if input_shape is None:
                img_sample = imread(os.path.join(os.path.split(
                    proto.dataset[1].uri)[0], row[0]), channel_first=True)
                if len(img_sample.shape) == 2:
                    input_shape = (1,) + img_sample.shape
                else:
                    input_shape = img_sample.shape

    train_optimizer = NNPSolverToNNASOptimizer(args, proto.optimizer[0].solver)
    if len(proto.optimizer) == 2:
        valid_optimizer = NNPSolverToNNASOptimizer(
            args, proto.optimizer[1].solver).copy()
        logger.info(
            99, f'"{proto.optimizer[0].name}" is used for train and "{proto.optimizer[1].name}" is used for valid.')
    else:
        valid_optimizer = train_optimizer.copy()
        if 'lr_scheduler' in valid_optimizer:
            del valid_optimizer['lr_scheduler']
        logger.info(
            99, f'Only a single optimizer was specified. "{proto.optimizer[0].name}" is used for both train and valid.')

    def progress(s):
        with open(progress_txt, 'w') as pt:
            pt.write(s)

    def convert_log(p):
        while p is not None:
            line = p.stdout.readline()
            s = line.decode().strip()
            if s and (s[0] == '['):
                progress(s)
            elif s:
                logger.log(99, s)
            if not line and p.poll() is not None:
                break
        progress('')

    # Create config file for search
    if args.mode == 'both' or args.mode == 'search':
        search_dir = os.path.join(args.output_dir, 'nnabla_nas_search')
        os.makedirs(search_dir)
        search_config_file = os.path.join(
            search_dir, 'nnabla_nas_config_darts_search.json')

        search_json = {
            "dataloader": {
                "csv": {
                    "train_portion": args.train_portion,
                    "train_file": proto.dataset[0].uri,
                    "valid_file": proto.dataset[1].uri,
                    "train_cache_dir": proto.dataset[0].cache_dir,
                    "valid_cache_dir": proto.dataset[0].cache_dir,
                    "augmentation": {
                        "type": "cifar10"  # args.augmentation
                    }
                }
            },
            "network": {
                "darts": {
                    "in_channels": input_shape[0],
                    "init_channels": args.init_channel_search,
                    "num_cells": args.cells_search,
                    "num_classes": num_class,
                    "shared": True,
                    "mode": "full"
                },
            },
            "optimizer": {
                "train": train_optimizer,
                "valid": valid_optimizer
            },
            "hparams": {
                "batch_size_train": proto.dataset[0].batch_size * proto.optimizer[0].update_interval,
                "batch_size_valid": proto.dataset[0].batch_size * proto.optimizer[0].update_interval,
                "mini_batch_train": proto.dataset[0].batch_size,
                "mini_batch_valid": proto.dataset[0].batch_size,
                "epoch": args.epoch_search,
                "print_frequency": args.print_frequency,
                "warmup": warmup_iter,
                "input_shapes": [input_shape],
                "target_shapes": [[1,]]
            }}
        with open(search_config_file, mode='w') as f:
            json.dump(search_json, f, indent=4)

        # Run nnabla nas (search)
        search_args = [
            'python',
            nnabla_nas_util.main.__file__,
            '-f',
            search_config_file,
            '-a',
            'DartsSearcher',
            '-s',
            '-o',
            search_dir,
            '--save-nnp',
            '--no-visualize']
        p = subprocess.Popen(search_args, stdout=subprocess.PIPE)
        convert_log(p)
        if p is None or p.poll() != 0:
            logger.critical('Search failed.')
            sys.exit(1)

        genotype = os.path.join(search_dir, 'arch.json')

        # move the trained model to the output_dir
        if args.mode == 'search':
            shutil.move(os.path.join(
                search_dir, 'results.nnp'), args.output_dir)

    if args.mode == 'both' or args.mode == 'train':
        # Create config file for training
        train_dir = os.path.join(args.output_dir, 'nnabla_nas_train')
        os.makedirs(train_dir)
        train_config_file = os.path.join(
            train_dir, 'nnabla_nas_config_darts_train.json')

        train_json = {
            "dataloader": {
                "csv": {
                    "train_file": proto.dataset[0].uri,
                    "valid_file": proto.dataset[1].uri,
                    "train_cache_dir": proto.dataset[0].cache_dir,
                    "valid_cache_dir": proto.dataset[0].cache_dir,
                    "augmentation": {
                        "type": "cifar10"  # args.augmentation
                    }
                }
            },
            "network": {
                "darts": {
                    "in_channels": input_shape[0],
                    "init_channels": args.init_channel_train,
                    "num_cells": args.cells_train,
                    "num_classes": num_class,
                    "auxiliary": args.auxiliary_train,
                    "drop_path": args.drop_path_train,
                    "genotype": genotype
                },
            },
            "optimizer": {
                "train": train_optimizer,
            },
            "hparams": {
                "batch_size_train": proto.dataset[0].batch_size * proto.optimizer[0].update_interval,
                "batch_size_valid": proto.dataset[0].batch_size * proto.optimizer[0].update_interval,
                "mini_batch_train": proto.dataset[0].batch_size,
                "mini_batch_valid": proto.dataset[0].batch_size,
                "epoch": proto.training_config.max_epoch,
                "print_frequency": args.print_frequency,
                "warmup": warmup_iter,
                "input_shapes": [input_shape],
                "target_shapes": [[1,]]
            }}
        with open(train_config_file, mode='w') as f:
            json.dump(train_json, f, indent=4)

        # Run nnabla nas (train)
        train_args = [
            'python',
            nnabla_nas_util.main.__file__,
            '-f',
            train_config_file,
            '-a',
            'Trainer',
            '-o',
            train_dir,
            '--save-nnp',
            '--no-visualize']
        p = subprocess.Popen(train_args, stdout=subprocess.PIPE)
        convert_log(p)
        if p is None or p.poll() != 0:
            logger.critical('Training failed.')
            sys.exit(1)

        # move the trained model to the output_dir
        shutil.move(os.path.join(train_dir, 'results.nnp'), args.output_dir)

    # create results.nntxt
    result_nnp = os.path.join(args.output_dir, 'results.nnp')
    result_nntxt = os.path.join(args.output_dir, 'results.nntxt')
    with zipfile.ZipFile(result_nnp) as z:
        for name in z.namelist():
            ext = os.path.splitext(name)[1].lower()
            if ext == ".nntxt" or ext == ".prototxt":
                with z.open(name, mode='r') as f:
                    with open(result_nntxt, 'wb') as w:
                        w.write(f.read())

    # edit nntxt option
    result_proto = nnabla_pb2.NNablaProtoBuf()
    with open(result_nntxt, mode='r') as f:
        text_format.Merge(f.read(), result_proto)
    result_proto.executor[0].no_image_normalization = True
    with open(result_nntxt, mode='w') as f:
        text_format.PrintMessage(result_proto, f)

    result_model_file_tmp = result_nnp + '.tmp'
    with zipfile.ZipFile(result_nnp, 'r') as zr:
        with zipfile.ZipFile(result_model_file_tmp, 'w') as zw:
            for info in zr.infolist():
                ext = os.path.splitext(info.filename)[1].lower()
                if ext == ".nntxt" or ext == ".prototxt":
                    zw.write(result_nntxt, arcname=info.filename)
                else:
                    bin = zr.read(info.filename)
                    zw.writestr(info, bin)
    shutil.move(result_model_file_tmp, result_nnp)

    logger.log(99, 'Training Completed.')


def main():
    parser = argparse.ArgumentParser(
        description='DARTS (image classification)\n' +
        '\n' +
        'DARTS: Differentiable Architecture Search\n' +
        'Hanxiao Liu, Karen Simonyan, Yiming Yang\n' +
        'https://arxiv.org/abs/1806.09055', formatter_class=argparse.RawTextHelpFormatter)
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
        help='training mode (option:both,search,train) default=both',
        default='both')
    parser.add_argument(
        '-ar',
        '--arch',
        help='arch.json file for training mode. not required when doing search(file)')
    parser.add_argument(
        '-p',
        '--train_portion',
        help='ratio of data used for search(float) default=0.5',
        type=float, default=0.5)
    '''
    # NNabla-NAS DARTS only supports 3,32,32 input
    parser.add_argument(
        '-a',
        '--augmentation',
        help='image augmentation mode (option:cifar10,imagenet) default=cifar10',
        required=True)
    '''
    parser.add_argument(
        '-es',
        '--epoch_search',
        help='number of epoch for search(int) default=50',
        type=int, default=50)
    parser.add_argument(
        '-cs',
        '--init_channel_search',
        help='number of init channel for search(int) default=16',
        type=int, default=16)
    parser.add_argument(
        '-ls',
        '--cells_search',
        help='number of cells for search(int) default=8',
        type=int, default=8)
    parser.add_argument(
        '-ct',
        '--init_channel_train',
        help='number of init channel for training(int) default=26',
        type=int, default=26)
    parser.add_argument(
        '-lt',
        '--cells_train',
        help='number of cells for training(int) default=20',
        type=int, default=20)
    parser.add_argument(
        '-at',
        '--auxiliary_train',
        help='use auxiliary loss for training(bool) default=True',
        action='store_true')
    parser.add_argument(
        '-pt',
        '--drop_path_train',
        help='ratio of drop path for training(float) default=0.1',
        type=float, default=0.1)
    parser.add_argument(
        '-f',
        '--print_frequency',
        help='frequency of log output by the number of iteration (int) default=20',
        type=int, default=20)
    parser.set_defaults(func=func)

    args = parser.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
