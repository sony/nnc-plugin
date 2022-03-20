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

<<<<<<< HEAD
<<<<<<< HEAD
import argparse
from distutils.util import strtobool
from .utils import ensure_dir


def add_finetune_args(parser):
    parser.add_argument(
        "--lmbd",
        "-lm",
        type=float,
        default=0.003,
        required=True,
        help="weight factor of l2, default=0.03",
    )

    parser.add_argument(
        "--epoch", "-ep", type=int, default=1000, required=True, help="finetune epochs, default=1000"
    )

    return parser


def add_feature_extract_args(parser):
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp'
    )

    parser.add_argument(
        "-tr",
        "--input-train",
        help="path to training dataset csv file (csv)",
        required=True,
    )

    parser.add_argument(
        "-v",
        "--input-val",
        help="path to validation dataset csv file (csv)",
        required=True,
    )

    parser.add_argument(
        "--normalize",
        "-n",
        action="store_true",
        default=False,
        help="Image Normaliztion (1.0/255.0) (bool)",
    )
    return parser


<<<<<<< HEAD
def get_basic_args(parser, monitor_path='representer_point_results'):
=======
=======
import argparse
from distutils.util import strtobool
from typing_extensions import Required
from .utils import ensure_dir
>>>>>>> a1273c6 (first commit)


<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Pretraining Model")
    parser.add_argument("--model_save_interval", type=int, default=30)
>>>>>>> fe1dfc3 (first commit)
=======
def add_finetune_args(parser):
    parser.add_argument("--lmbd",
                        type=float,
                        default=0.003,
                        requiered=True,
                        help='weight factor of l2')
    parser.add_argument("--epoch",
                        type=int,
                        default=1000,
                        required=True,
                        help='finetune epochs')

    return parser


def add_feature_extract_args(parser):
    parser.add_argument('--model-path',
                        '-mp',
                        type=str,
                        help='pretrained model nnp file (file)',
                        required=True)

    parser.add_argument('-tr',
                        '--input-train',
                        help='path to training dataset csv file (csv)',
                        required=True)

    parser.add_argument('-v',
                        '--input-val',
                        help=argparse.SUPPRESS,
                        required=False)

    parser.add_argument('--normalize',
                        '-n',
                        type=strtobool,
                        default=False,
                        required=True,
                        help='Image Normaliztion (1.0/255.0) (bool)')
    return parser


def get_basic_args(parser, monitor_path='representer_point_results'):
>>>>>>> a1273c6 (first commit)
=======
def get_basic_args(parser, monitor_path="representer_point_results"):
>>>>>>> b80d247 (bug fix)
    parser.add_argument(
        "--shuffle_label",
        type=strtobool,
        default=True,
        help="whether shuffle trainig label or not",
    )
<<<<<<< HEAD
<<<<<<< HEAD
=======
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="if resume training, you put your weight",
    )
>>>>>>> fe1dfc3 (first commit)
=======
>>>>>>> a1273c6 (first commit)

    parser.add_argument("-top", "--top_k", type=int, default=3, required=True,
                        help="top-k influenced samples are presented, default=3")

    parser.add_argument(
        "-N",
        "--num-samples",
        type=int,
        default=10,
        help="Number of test samples for influence analysis, default=10"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path to output csv file (csv) default=influence_sample.csv",
        default="influence_samples.csv",
        required=True,
    )

    parser.add_argument(
        "--device_id",
        "-d",
        type=str,
        default="0",
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        help=
        "Device ID the training run on. This is only valid if you specify `-c cudnn`.",
=======
        help="Device ID the training run on. This is only valid if you specify `-c cudnn`.",
>>>>>>> fe1dfc3 (first commit)
=======
        help=
        "Device ID the training run on. This is only valid if you specify `-c cudnn`.",
>>>>>>> a1273c6 (first commit)
=======
        required=False,
<<<<<<< HEAD
        help="Device ID the training run on. This is only valid if you specify `-c cudnn`.",
>>>>>>> b80d247 (bug fix)
=======
        help=argparse.SUPPRESS,
>>>>>>> 704bb9a (edit detailed argparse)
    )

    parser.add_argument(
        "--type_config",
        "-t",
        type=str,
        default="float",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--context",
        "-c",
        type=str,
        default="cudnn",
        help=argparse.SUPPRESS,
    )

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    parser.add_argument("--monitor-path", type=str, default=monitor_path)

    parser.add_argument(
        "--batch-size",
=======
    parser.add_argument(
        "--train_batch_size",
>>>>>>> fe1dfc3 (first commit)
=======
    parser.add_argument("--monitor-path", type=str, default=monitor_path)
=======
    parser.add_argument(
        "--monitor-path", type=str, help=argparse.SUPPRESS, default=monitor_path
    )
>>>>>>> b80d247 (bug fix)

    parser.add_argument(
        "--batch-size",
>>>>>>> a1273c6 (first commit)
        type=int,
        default=100,
        help=argparse.SUPPRESS,
    )
<<<<<<< HEAD
<<<<<<< HEAD
    return parser


def get_infl_args():
    parser = argparse.ArgumentParser(
        description="Representer Point\n"
        + "\n"
        + '"Representer Point Selection for Explaining Deep Neural Networks"\n'
        + "Chih-Kuan Yeh, Joon Sik Kim, Ian E.H. Yen, Pradeep Ravikumar. (2018)\n"
        + "https://arxiv.org/abs/1811.09720\n"
        + "",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser = get_basic_args(parser)
    parser = add_feature_extract_args(parser)
    parser = add_finetune_args(parser)
    args = parser.parse_args()
    ensure_dir(args.monitor_path)
=======
=======
    return parser
>>>>>>> a1273c6 (first commit)


def get_infl_args():
    parser = argparse.ArgumentParser(
        description='Representer Point\n' + '\n' +
        '"Representer Point Selection for Explaining Deep Neural Networks"\n' +
        'Chih-Kuan Yeh, Joon Sik Kim, Ian E.H. Yen, Pradeep Ravikumar. (2018)\n'
        + 'https://arxiv.org/abs/1811.09720\n' + '',
        formatter_class=argparse.RawTextHelpFormatter)

    parser = get_basic_args(parser)
    parser = add_feature_extract_args(parser)
    parser = add_finetune_args(parser)
    args = parser.parse_args()
<<<<<<< HEAD

>>>>>>> fe1dfc3 (first commit)
=======
    # ensure_dir(args.model_save_path)
>>>>>>> a1273c6 (first commit)
    return args
