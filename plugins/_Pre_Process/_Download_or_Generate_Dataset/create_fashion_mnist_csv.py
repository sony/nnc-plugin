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
import argparse
import csv
from nnabla.utils.image_utils import imsave
from contextlib import contextmanager
import numpy
import struct
import zlib
import tqdm

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download, get_data_home


class FashionMnistDataSource(DataSource):
    '''
    Get data directly from Fashion-MNIST dataset from Internet(https://github.com/zalandoresearch/fashion-mnist).

    Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
    Han Xiao, Kashif Rasul, Roland Vollgraf.
    arXiv:1708.07747
    '''

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, train=True, shuffle=False, rng=None):
        super(FashionMnistDataSource, self).__init__(shuffle=shuffle)
        self._train = train
        if self._train:
            image_uri = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
            label_uri = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz'
        else:
            image_uri = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz'
            label_uri = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'

        logger.info('Getting label data from {}.'.format(label_uri))

        r = download(label_uri, output_file=os.path.join(
            get_data_home(), 'fashion-mnist-' + label_uri.split('/')[-1]))
        data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
        _, size = struct.unpack('>II', data[0:8])
        self._labels = numpy.frombuffer(data[8:], numpy.uint8).reshape(-1, 1)
        r.close()
        logger.info('Getting label data done.')

        logger.info('Getting image data from {}.'.format(image_uri))
        r = download(image_uri, output_file=os.path.join(
            get_data_home(), 'fashion-mnist-' + image_uri.split('/')[-1]))
        data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
        _, size, height, width = struct.unpack('>IIII', data[0:16])
        self._images = numpy.frombuffer(data[16:], numpy.uint8).reshape(
            size, 1, height, width)
        r.close()
        logger.info('Getting image data done.')

        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = numpy.random.RandomState(313)
        self.rng = rng
        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = numpy.arange(self._size)
        super(FashionMnistDataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()


@contextmanager
def data_iterator_fashion_mnist(batch_size,
                                train=True,
                                rng=None,
                                shuffle=True,
                                with_memory_cache=False,
                                with_file_cache=False):
    '''
    Provide DataIterator with :py:class:`FashionMnistDataSource`
    with_memory_cache, with_parallel and with_file_cache option's default value is all False,
    because :py:class:`FashionMnistDataSource` is able to store all data into memory.

    For example,

    .. code-block:: python

        with data_iterator_fashion_mnist(True, batch_size) as di:
            for data in di:
                SOME CODE TO USE data.

    '''
    with FashionMnistDataSource(train=train, shuffle=shuffle, rng=rng) as ds, \
        data_iterator(ds,
                      batch_size,
                      rng=None,
                      with_memory_cache=with_memory_cache,
                      with_file_cache=with_file_cache) as di:
        yield di


def data_iterator_to_csv(csv_path, csv_file_name, data_path, data_iterator):
    index = 0
    csv_data = []
    labels = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress',
              'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    with data_iterator as data:
        line = ['x:image', 'y:label;' + ';'.join(labels)]
        csv_data.append(line)
        pbar = tqdm.tqdm(total=data.size, unit='images')
        initial_epoch = data.epoch
        while data.epoch == initial_epoch:
            d = data.next()
            for i in range(len(d[0])):
                label = d[1][i][0]
                file_name = data_path + \
                    '/{}'.format(labels[label]) + '/{}.png'.format(index)
                full_path = os.path.join(
                    csv_path, file_name.replace('/', os.path.sep))
                directory = os.path.dirname(full_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                imsave(full_path, d[0][i].reshape(28, 28))
                csv_data.append([file_name, label])
                index += 1
                pbar.update(1)
        pbar.close()
    with open(os.path.join(csv_path, csv_file_name), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)
    return csv_data


def func(args):
    path = args.output_dir

    # Create original training set
    logger.log(99, 'Downloading Fashion-MNIST training set images...')
    train_di = data_iterator_fashion_mnist(60000, True, None, False)
    logger.log(99, 'Creating "fashion_mnist_training.csv"... ')
    train_csv = data_iterator_to_csv(
        path, 'fashion_mnist_training.csv', './training', train_di)

    # Create original test set
    logger.log(99, 'Downloading Fashion-MNIST test set images...')
    validation_di = data_iterator_fashion_mnist(10000, False, None, False)
    logger.log(99, 'Creating "fashion_mnist_test.csv"... ')
    test_csv = data_iterator_to_csv(
        path, 'fashion_mnist_test.csv', './validation', validation_di)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='FashionMNIST\n\n' +
        'Download Fashion-MNIST dataset from https://github.com/zalandoresearch/fashion-mnist.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=FashionMNIST',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
