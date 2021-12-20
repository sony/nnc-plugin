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

"""
Provide data iterator for MNIST examples.
"""
import argparse
import random
import os
import numpy
import struct
import zlib
import tqdm
import numpy as np
import csv

from imageio import imwrite
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)


def load_mnist(train=True):
    """
    Load MNIST dataset images and labels from the original page by Yan LeCun or the cache file.

    Args:
        train (bool): The testing dataset will be returned if False. Training data has 60000 images, while testing has 10000 images.

    Returns:
        numpy.ndarray: A shape of (#images, 1, 28, 28). Values in [0.0, 1.0].
        numpy.ndarray: A shape of (#images, 1). Values in {0, 1, ..., 9}.

    """
    if train:
        image_uri = "https://github.com/zalandoresearch/fashion-mnist/train-images-idx3-ubyte.gz"
        label_uri = "https://github.com/zalandoresearch/fashion-mnist/train-labels-idx1-ubyte.gz"
    else:
        image_uri = (
            "https://github.com/zalandoresearch/fashion-mnist/t10k-images-idx3-ubyte.gz"
        )
        label_uri = (
            "https://github.com/zalandoresearch/fashion-mnist/t10k-labels-idx1-ubyte.gz"
        )
    logger.info("Getting label data from {}.".format(label_uri))
    # With python3 we can write this logic as following, but with
    # python2, gzip.object does not support file-like object and
    # urllib.request does not support 'with statement'.
    #
    #   with request.urlopen(label_uri) as r, gzip.open(r) as f:
    #       _, size = struct.unpack('>II', f.read(8))
    #       labels = numpy.frombuffer(f.read(), numpy.uint8).reshape(-1, 1)
    #
    r = download(label_uri)
    data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
    _, size = struct.unpack(">II", data[0:8])
    labels = numpy.frombuffer(data[8:], numpy.uint8).reshape(-1, 1)
    r.close()
    logger.info("Getting label data done.")

    logger.info("Getting image data from {}.".format(image_uri))
    r = download(image_uri)
    data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
    _, size, height, width = struct.unpack(">IIII", data[0:16])
    images = numpy.frombuffer(data[16:], numpy.uint8).reshape(
        size, 1, height, width)
    r.close()
    logger.info("Getting image data done.")

    return images, labels


class FashionMnistDataSource(DataSource):
    """
    Get data directly from MNIST dataset from Internet(yann.lecun.com).
    """

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        if args.label_shuffle and self._train:
            shuffle = self._shuffle_label[self._indexes[position]]
            return (image, label, shuffle)
        else:
            return (image, label)

    def __init__(
        self,
        train=True,
        shuffle=False,
        rng=None,
        label_shuffle=False,
        label_shuffle_rate=0.1,
    ):
        super(FashionMnistDataSource, self).__init__(shuffle=shuffle)
        self._train = train

        self._images, self._labels = load_mnist(train)
        if label_shuffle:
            raw_label = self._labels.copy()
            self.shuffle_rate = label_shuffle_rate
            self._shuffle_label = self.label_shuffle()
            print(f"{self.shuffle_rate*100}% of data was shuffled ")
            print(
                "shuffle_label_number: ", len(
                    np.where(self._labels != self._shuffle_label)[0]),
            )
        self._size = self._labels.size
        if args.label_shuffle and train:
            self._variables = ("x", "y", "shuffle")
        else:
            self._variables = ("x", "y")
        if rng is None:
            rng = numpy.random.RandomState(313)
        self.rng = rng
        self.reset()

    def label_shuffle(self):
        num_cls = int(np.max(self._labels)) + 1
        shuffle_label = self._labels.copy()
        extract_num = int(len(self._labels) * self.shuffle_rate // num_cls)
        for i in range(num_cls):
            extract_ind = np.where(self._labels == i)[0]
            labels = [j for j in range(num_cls)]
            labels.remove(i)  # candidate of shuffle label
            artificial_label = [
                labels[int(i) % (num_cls - 1)] for i in range(int(extract_num))
            ]
            artificial_label = np.array(
                random.sample(artificial_label, len(artificial_label))
            ).astype("float32")
            convert_label = np.array([i for _ in range(len(extract_ind))])
            convert_label[-extract_num:] = artificial_label
            random.shuffle(convert_label)

            shuffle_label[extract_ind] = convert_label.reshape(-1, 1)

        return shuffle_label

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

    @property
    def shuffle_labels(self):
        return self._shuffle_label.copy()


def data_iterator_fashion_mnist(
    batch_size,
    train=True,
    rng=None,
    shuffle=True,
    with_memory_cache=False,
    with_file_cache=False,
    label_shuffle=False,
    label_shuffle_rate=0.1,
):
    """
    Provide DataIterator with :py:class:`FashionMnistDataSource`
    with_memory_cache and with_file_cache option's default value is all False,
    because :py:class:`FashionMnistDataSource` is able to store all data into memory.

    For example,

    .. code-block:: python

        with data_iterator_mnist(True, batch_size) as di:
            for data in di:
                SOME CODE TO USE data.

    """
    return data_iterator(
        FashionMnistDataSource(
            train=train,
            shuffle=shuffle,
            rng=rng,
            label_shuffle=label_shuffle,
            label_shuffle_rate=label_shuffle_rate,
        ),
        batch_size,
        rng,
        with_memory_cache,
        with_file_cache,
    )


def data_iterator_to_csv(csv_path, csv_file_name, data_path, data_iterator, shuffle):
    index = 0
    csv_data = []
    labels = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle_boot",
    ]

    with data_iterator as data:
        if shuffle:
            line = ["x:image", "y:label;" + ";".join(labels), "original_label"]
        else:
            line = ["x:image", "y:label;" + ";".join(labels)]
        csv_data.append(line)
        pbar = tqdm.tqdm(total=data.size, unit="images")
        initial_epoch = data.epoch
        while data.epoch == initial_epoch:
            d = data.next()
            for i in range(len(d[0])):
                label = d[1][i][0]
                file_name = (
                    data_path +
                    "/{}".format(labels[label]) + "/{}.png".format(index)
                )
                full_path = os.path.join(
                    csv_path, file_name.replace("/", os.path.sep))
                directory = os.path.dirname(full_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                imwrite(full_path, d[0][i].reshape(28, 28))
                if shuffle:
                    shuffled_label = d[2][i][0]
                    csv_data.append([file_name, shuffled_label, label])
                else:
                    csv_data.append([file_name, label])
                index += 1
                pbar.update(1)
        pbar.close()
    with open(os.path.join(csv_path, csv_file_name), "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(csv_data)
    return csv_data


def main():
    path = os.path.abspath(os.path.dirname(__file__))

    # Create original training set
    logger.log(99, "Downloading Fashion MNIST dataset...")

    train_di = data_iterator_fashion_mnist(
        60000,
        True,
        None,
        False,
        label_shuffle=args.label_shuffle,
        label_shuffle_rate=args.shuffle_rate,
    )
    if args.label_shuffle:
        logger.log(99, 'Creating "fashion_mnist_training_shuffle.csv"... ')
        train_csv = data_iterator_to_csv(
            path,
            "fashion_mnist_training_shuffle.csv",
            os.path.join(os.getcwd(), "training"),
            train_di,
            shuffle=args.label_shuffle,
        )
    else:
        logger.log(99, 'Creating "fashion_mnist_training.csv"... ')
        train_csv = data_iterator_to_csv(
            path,
            "fashion_mnist_training.csv",
            os.path.join(os.getcwd(), "training"),
            train_di,
            shuffle=False,
        )

    # Create original test set
    validation_di = data_iterator_fashion_mnist(10000, False, None, False)
    logger.log(99, 'Creating "fashion_mnist_test.csv"... ')
    test_csv = data_iterator_to_csv(
        path,
        "fashion_mnist_test.csv",
        os.path.join(os.getcwd(), "validation"),
        validation_di,
        shuffle=False,
    )

    logger.log(99, "Dataset creation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_shuffle", action="store_true", help="generate label shuffled dataset"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_rate", type=float, default=0.1)
    args = parser.parse_args()

    set_seed()
    print("Label Shuffle: ", args.label_shuffle)
    main()
