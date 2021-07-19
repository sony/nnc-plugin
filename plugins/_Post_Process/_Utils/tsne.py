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
import tqdm
from sklearn.manifold import TSNE
from nnabla import logger
from nnabla.utils.data_iterator import data_iterator_csv_dataset


def func(args):
    # Load variable
    data_iterator = (lambda: data_iterator_csv_dataset(
        uri=args.input,
        batch_size=64,
        shuffle=False,
        normalize=True,
        with_memory_cache=False,
        with_file_cache=False))

    logger.log(99, 'Loading variable...')
    dataset = []
    with data_iterator() as di:
        pbar = tqdm.tqdm(total=di.size)
        while len(dataset) < di.size:
            data = di.next()
            variable = data[di.variables.index(args.variable)]
            dataset.extend(variable)
            pbar.update(len(variable))
        pbar.close()

    dataset = np.array(dataset)[:di.size].reshape(di.size, -1)
    logger.log(99, 'variable={}, length={}, dim={}'.format(
        args.variable, dataset.shape[0], dataset.shape[1]))

    # t-SNE
    logger.log(99, 'Processing t-SNE...')
    dim = int(args.dim)
    result = TSNE(n_components=dim, random_state=0).fit_transform(dataset)

    # output
    with open(args.input, newline='', encoding='utf-8-sig') as f:
        rows = [row for row in csv.reader(f)]
    row0 = rows.pop(0)

    row0.extend([args.variable + '_tsne__{}'.format(i) for i in range(dim)])
    for i, y in enumerate(result):
        rows[i].extend(y)
    with open(args.output, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(row0)
        writer.writerows(rows)

    logger.log(99, 't-SNE completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='t-SNE\n\n' +
        'L. van der Maaten, G. Hinton. Visualizing Data using t-SNE\n' +
        'http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i',
        '--input',
        help='path to input csv file (csv) default=output_result.csv',
        required=True,
        default='output_result.csv')
    parser.add_argument(
        '-v',
        '--variable',
        help="Variable to be processed (variable) default=x",
        required=True,
        default="x")
    parser.add_argument(
        '-d',
        '--dim',
        help='dimension of the embedded space (variable) default=2',
        default=2)
    parser.add_argument(
        '-o',
        '--output',
        help='path to output csv file (csv) default=tsne.csv',
        required=True,
        default='tsne.csv')
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
