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
Provide data iterator for CIFAR10 examples.
"""
from contextlib import contextmanager
import argparse
import numpy as np
import random
import tarfile
import os
import tqdm
import csv
from imageio import imwrite

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)


class Cifar10DataSource(DataSource):
    """
    Get data directly from cifar10 dataset from Internet(yann.lecun.com).
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
        super(Cifar10DataSource, self).__init__(shuffle=shuffle)

        self._train = train
        data_uri = "https://dl.sony.com/app/datasets/cifar-10-python.tar.gz"
        logger.info("Getting labeled data from {}.".format(data_uri))
        r = download(data_uri)  # file object returned
        with tarfile.open(fileobj=r, mode="r:gz") as fpin:
            # Training data
            if train:
                images = []
                labels = []
                for member in fpin.getmembers():
                    if "data_batch" not in member.name:
                        continue
                    fp = fpin.extractfile(member)
                    data = np.load(fp, allow_pickle=True, encoding="bytes")
                    images.append(data[b"data"])
                    labels.append(data[b"labels"])
                self._size = 50000
                self._images = np.concatenate(
                    images).reshape(self._size, 3, 32, 32)
                self._labels = np.concatenate(labels).reshape(-1, 1)
                if label_shuffle:
                    self.shuffle_rate = label_shuffle_rate
                    self._shuffle_label = self.label_shuffle()
                    print(f"{self.shuffle_rate*100}% of data was shuffled ")
                    print(
                        "shuffle_label_number: ",
                        len(np.where(self._labels != self._shuffle_label)[0]),
                    )
            # Validation data
            else:
                for member in fpin.getmembers():
                    if "test_batch" not in member.name:
                        continue
                    fp = fpin.extractfile(member)
                    data = np.load(fp, allow_pickle=True, encoding="bytes")
                    images = data[b"data"]
                    labels = data[b"labels"]
                self._size = 10000
                self._images = images.reshape(self._size, 3, 32, 32)
                self._labels = np.array(labels).reshape(-1, 1)
        r.close()
        logger.info("Getting labeled data from {}.".format(data_uri))

        self._size = self._labels.size
        if args.label_shuffle and train:
            self._variables = ("x", "y", "shuffle")
        else:
            self._variables = ("x", "y")
        if rng is None:
            rng = np.random.RandomState(313)
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
            self._indexes = np.arange(self._size)
        super(Cifar10DataSource, self).reset()

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


@contextmanager
def data_iterator_cifar10(
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
    Provide DataIterator with :py:class:`Cifar10DataSource`
    with_memory_cache, with_parallel and with_file_cache option's default value is all False,
    because :py:class:`Cifar10DataSource` is able to store all data into memory.

    For example,

    .. code-block:: python

        with data_iterator_cifar10(True, batch_size) as di:
            for data in di:
                SOME CODE TO USE data.

    """
    with Cifar10DataSource(
        train=train,
        shuffle=shuffle,
        rng=rng,
        label_shuffle=label_shuffle,
        label_shuffle_rate=label_shuffle_rate,
    ) as ds, data_iterator(
        ds,
        batch_size,
        rng=rng,
        with_memory_cache=with_memory_cache,
        with_file_cache=with_file_cache,
    ) as di:
        yield di


def data_iterator_to_csv(csv_path, csv_file_name, data_path, data_iterator, shuffle):
    index = 0
    csv_data = []
    labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
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
                imwrite(full_path, d[0][i].reshape(
                    3, 32, 32).transpose(1, 2, 0))
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
    logger.log(99, "Downloading CIFAR-10 dataset...")

    train_di = data_iterator_cifar10(
        50000,
        True,
        None,
        False,
        label_shuffle=args.label_shuffle,
        label_shuffle_rate=args.shuffle_rate,
    )
    logger.log(99, 'Creating "cifar10_training.csv"... ')
    if args.label_shuffle:
        train_csv = data_iterator_to_csv(
            path,
            "cifar10_training_shuffle.csv",
            "./training",
            train_di,
            shuffle=args.label_shuffle,
        )
    else:
        train_csv = data_iterator_to_csv(
            path,
            "cifar10_training.csv",
            "./training",
            train_di,
            shuffle=False,
        )

    # Create original test set
    validation_di = data_iterator_cifar10(10000, False, None, False)
    logger.log(99, 'Creating "cifar10_test.csv"... ')
    test_csv = data_iterator_to_csv(
        path,
        "cifar10_test.csv",
        "./validation",
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
