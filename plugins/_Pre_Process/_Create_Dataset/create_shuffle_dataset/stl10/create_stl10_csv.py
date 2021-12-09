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
import sys
import tarfile
import errno
import numpy
import tqdm
import numpy as np
import csv
import cv2

from glob import glob

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib
from imageio import imwrite
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource


DATA_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
DATA_DIR = "./data"


def save_image(image, name):
    # imsave("%s.png" % name, image, format="png")
    imwrite("%s.png" % name, image, format="png")


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, "rb") as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """
    with open(path_to_data, "rb") as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def save_images(images, labels, root_dir):
    print("Saving images to disk")
    i = 0
    for image in images:
        label = labels[i]
        directory = os.path.join(root_dir, str(label - 1))
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = os.path.join(directory, str(i))
        save_image(image, filename)
        i = i + 1


def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split("/")[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                "\rDownloading %s %.2f%%"
                % (filename, float(count * block_size) / float(total_size) * 100.0)
            )
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(
            DATA_URL, filepath, reporthook=_progress)
        print("Downloaded", filename)
        tarfile.open(filepath, "r:gz").extractall(dest_directory)


class Stl10DataSource(DataSource):
    def __init__(
        self,
        train=True,
        dataset_size=5000,
        cls=10,
        shuffle=False,
        rng=None,
        label_shuffle=True,
        label_shuffle_rate=0.1,
        seed=0,
        size=96,
    ):
        super(Stl10DataSource, self).__init__(shuffle=shuffle)

        self._train = train
        self.seed = seed
        self.cls = cls
        self.label_shuffle_ = label_shuffle
        self._image_size = size
        if train:
            self._size = dataset_size
            data_root = "./training"
            cls_dirs = glob(os.path.join(data_root, "*"))
            self.path, images, labels = self.loader(cls_dirs)

            self._labels = np.array(labels).reshape(-1, 1)
            img_np = np.concatenate(images, axis=0).reshape(
                5000, self._image_size, self._image_size, 3
            )
            self._images = np.transpose(img_np, (0, 3, 1, 2))

            self._shuffle_labels = self._labels.copy()
            if label_shuffle:
                self.shuffle_rate = label_shuffle_rate
                self.label_shuffle()
                print(f"{self.shuffle_rate*100}% of data was shuffled ")
                self._variables = ("path", "x", "y", "shuffle")
            else:
                self._variables = ("path", "x", "y")
        # Validation data
        else:
            self._size = 8000
            data_root = "./validation"
            cls_dirs = glob(os.path.join(data_root, "*"))
            self.path, images, labels = self.loader(cls_dirs)
            self._labels = np.array(labels).reshape(-1, 1)
            img_np = np.concatenate(images, 0).reshape(
                8000, self._image_size, self._image_size, 3
            )
            self._images = np.transpose(img_np, (0, 3, 1, 2))
            print(self._images.shape)

            self._shuffle_labels = self._labels.copy()
            self._variables = ("path", "x", "y")

        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

    def loader(self, cls_dirs):
        images, labels = [], []
        img_paths = []
        for cls_dir in cls_dirs:
            cls_num = os.path.basename(cls_dir)
            img_path = glob(os.path.join(cls_dir, "*.png"))
            img_paths.extend(img_path)
            labels.extend([int(cls_num) for _ in range(len(img_path))])
            for img in img_path:
                if self._image_size != 96:
                    im = np.expand_dims(
                        cv2.resize(
                            cv2.imread(img), dsize=(self._image_size, self._image_size)
                        ),
                        0,
                    )
                else:
                    im = np.expand_dims(cv2.imread(img), 0)
                images.append(im)

        return img_paths, images, labels

    def _get_data(self, position):
        path = self.path[self._indexes[position]]
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        if (self._train) and (self.label_shuffle_):
            shuffle_label = self._shuffle_labels[self._indexes[position]]
            return (path, image, label, shuffle_label)
        else:
            return (path, image, label)

    def label_shuffle(self):
        random.seed(self.seed)
        num_cls = int(np.max(self._labels)) + 1
        raw_label = self._labels.copy()
        extract_num = int(len(self._labels) * self.shuffle_rate // self.cls)
        for i in range(num_cls):
            extract_ind = np.where(raw_label == i)[0]
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

            self._shuffle_labels[extract_ind] = convert_label.reshape(-1, 1)

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(Stl10DataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def raw_labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()

    @property
    def shuffled_labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        if self._shuffle_labels:
            return self._shuffle_labels.copy()


def data_iterator_stl10(
    batch_size,
    train=True,
    rng=None,
    shuffle=True,
    with_memory_cache=False,
    with_file_cache=False,
    label_shuffle=False,
    label_shuffle_rate=0.1,
    seed=0,
    size=96,
):
    """
    Provide DataIterator with :py:class:`Stl10DataSource`
    """
    return data_iterator(
        Stl10DataSource(
            train=train,
            cls=10,
            shuffle=shuffle,
            rng=rng,
            label_shuffle=label_shuffle,
            label_shuffle_rate=label_shuffle_rate,
            seed=seed,
            size=size,
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
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
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
                label = d[2][i][0]
                file_name = d[0][i]

                if shuffle:
                    shuffled_label = d[3][i][0]
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
    logger.log(99, "Downloading STL10 dataset...")

    DATA_PATH = "./data/stl10_binary/train_X.bin"
    LABEL_PATH = "./data/stl10_binary/train_y.bin"

    if not os.path.exists("./training"):
        download_and_extract()
        images = read_all_images(DATA_PATH)
        labels = read_labels(LABEL_PATH)
        save_images(images, labels, "./training")

    train_di = data_iterator_stl10(
        5000,
        True,
        None,
        False,
        label_shuffle=args.label_shuffle,
        label_shuffle_rate=args.shuffle_rate,
    )
    if args.label_shuffle:
        logger.log(99, 'Creating "stl10_training_shuffle.csv"... ')
        train_csv = data_iterator_to_csv(
            path,
            "stl10_training_shuffle.csv",
            os.path.join(os.getcwd(), "training"),
            train_di,
            shuffle=args.label_shuffle,
        )
    else:
        logger.log(99, 'Creating "stl10_training.csv"... ')
        train_csv = data_iterator_to_csv(
            path,
            "stl10_training.csv",
            os.path.join(os.getcwd(), "training"),
            train_di,
            shuffle=False,
        )

    # Create original test set
    DATA_PATH = "./data/stl10_binary/test_X.bin"
    LABEL_PATH = "./data/stl10_binary/test_y.bin"

    if not os.path.exists("./validation"):
        download_and_extract()
        images = read_all_images(DATA_PATH)
        labels = read_labels(LABEL_PATH)
        save_images(images, labels, os.path.join(os.getcwd(), "validation"))

    validation_di = data_iterator_stl10(8000, False, None, False)
    logger.log(99, 'Creating "stl10_test.csv"... ')
    test_csv = data_iterator_to_csv(
        path,
        "stl10_test.csv",
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
