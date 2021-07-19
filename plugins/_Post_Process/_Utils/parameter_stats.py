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
import argparse
import csv
import numpy as np

import nnabla as nn
import nnabla.utils.load as load
from nnabla import logger


def func(args):
    # Load model
    info = load.load([args.model], prepare_data_iterator=False, batch_size=1)

    with open(args.output, 'w', newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'shape', 'size', 'max', 'min',
                         'max_abs', 'min_abs', 'mean', 'stdev'])
        for name, param in nn.get_parameters().items():
            writer.writerow([
                name,
                str(param.shape).replace('(', '').replace(')',
                                                          '').replace(', ', ' x ').replace(',', ''),
                param.size,
                np.max(param.d),
                np.min(param.d),
                np.max(np.abs(param.d)),
                np.min(np.abs(param.d)),
                np.mean(param.d),
                np.std(param.d)])

    logger.log(99, 'Parameter stats completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='Parameter Stats\n\nCalculate various statistics of parameter\n\n'
        '(size, max, min, max_abs, min_abs, mean, stdev)\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m',
        '--model',
        help='path to model nnp file (model) default=results.nnp',
        required=True,
        default='results.nnp')
    parser.add_argument(
        '-o',
        '--output',
        help='path to output csv file (csv) default=parameter_stats.csv',
        required=True,
        default='parameter_stats.csv')
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
