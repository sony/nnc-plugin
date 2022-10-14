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
import argparse
from face_evaluation_utils.face_calc_features import make_ita_dict
from nnabla import logger


def func(args):
    # parameters
    input_train = args.input_train
    output_name = args.output

    try:
        make_ita_dict(input_train, output_name)
        logger.log(99, "face evaluation completed successfully.")
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='face evaluation\n'
        + '', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-t', '--input-train', help='path to training dataset csv file (csv)', required=True)
    parser.add_argument(
        '-o', '--output', help='path to output csv file (csv) default=ita.csv', default='ita.csv')
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
