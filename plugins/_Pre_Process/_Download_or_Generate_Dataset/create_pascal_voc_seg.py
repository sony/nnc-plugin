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
import tarfile
import tqdm
import numpy as np
from PIL import Image, ImageOps
from nnabla.utils.image_utils import imresize, imsave

from nnabla.logger import logger
from nnabla.utils.data_source_loader import download


def convert_image(args):
    def pad_image(im):
        h = im.shape[0]
        w = im.shape[1]

        h1 = np.min([500, h + 8])
        w1 = np.min([500, w + 8])

        pad = (((h1 - h) // 2, h1 - h - (h1 - h) // 2),
               ((w1 - w) // 2, w1 - w - (w1 - w) // 2))
        if len(im.shape) == 3:
            pad = pad + ((0, 0),)
        im = np.pad(im, pad, 'constant', constant_values=255)

        pad = (((500 - h1) // 2, 500 - h1 - (500 - h1) // 2),
               ((500 - w1) // 2, 500 - w1 - (500 - w1) // 2))
        if len(im.shape) == 3:
            pad = pad + ((0, 0),)
        im = np.pad(im, pad, 'constant', constant_values=0)
        assert(im.shape[0] == 500 and im.shape[1] == 500)

        return im

    tar_file = args[0]
    img_id = args[1]
    target_dir = args[2]

    # open source image
    try:
        # open source image
        img_file_info = tar_file.getmember(
            "VOCdevkit/VOC2012/JPEGImages/" + img_id + ".jpg")
        with tar_file.extractfile(img_file_info) as f:
            img = np.asarray(Image.open(f))
        assert(len(img.shape) == 3)
        assert(img.shape[2] == 3)
        assert(img.shape[0] <= 500 and img.shape[1] <= 500)

        # open source label
        label_file_info = tar_file.getmember(
            "VOCdevkit/VOC2012/SegmentationClass/" + img_id + ".png")
        with tar_file.extractfile(label_file_info) as f:
            label = np.asarray(Image.open(f))
        assert(len(label.shape) == 2)
        assert(label.shape[0] <= 500 and label.shape[1] <= 500)

        # save 500px image
        img = pad_image(img)
        imsave(os.path.join(target_dir, 'images_500px', img_id + '.png'), img)

        # save 500px label
        label = pad_image(label)
        imsave(os.path.join(target_dir, 'labels_500px', img_id + '.png'), label)

        # save 125px image
        img = imresize(img, size=(125, 125))
        imsave(os.path.join(target_dir, 'images_125px', img_id + '.png'), img)

        # save 125px label
        label = imresize(label, size=(125, 125), interpolate="nearest")
        imsave(os.path.join(target_dir, 'labels_125px', img_id + '.png'), label)
    except:
        logger.warning(
            "Failed to convert %s." % (img_id))
        raise


def func(args):
    path = args.output_dir

    # Download PASCAL VOC 2012 dataset
    logger.log(99, 'Downloading PASCAL VOC 2012 dataset...')
    dataset_file = download(
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")

    # Open tar file
    logger.log(99, 'Converting images ...')
    tar_file = tarfile.open(fileobj=dataset_file)

    # Enumerate images　for segmentation
    file_info = tar_file.getmember(
        "VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt")
    f = tar_file.extractfile(file_info)
    img_ids = [id.decode('utf-8').strip() for id in f.readlines()]
    f.close()

    # Prepare output dirs
    def makedirs(path):
        try:
            os.makedirs(path)
        except:
            pass

    makedirs(os.path.join(path, 'images_500px'))
    makedirs(os.path.join(path, 'images_125px'))
    makedirs(os.path.join(path, 'labels_500px'))
    makedirs(os.path.join(path, 'labels_125px'))

    # Convert images
    for img_id in tqdm.tqdm(img_ids):
        # Convert jpeg image
        convert_image([tar_file, img_id, path])

    # Create dataset CSV files
    for set in ["train", "trainval", "val"]:
        file_info = tar_file.getmember(
            "VOCdevkit/VOC2012/ImageSets/Segmentation/" + set + ".txt")
        f = tar_file.extractfile(file_info)
        img_ids = [id.decode('utf-8').strip() for id in f.readlines()]
        f.close()
        for resolution in ["500px", "125px"]:
            csv_file_name = "pascal_voc_2012_seg_" + set + "_" + resolution + ".csv"
            logger.log(99, 'Creating ' + csv_file_name + ' ...')
            csv_data = [['x:image', 'y:label']]
            for img_id in img_ids:
                fn_str = resolution + "/" + img_id + ".png"
                csv_data.append(["./images_" + fn_str, "./labels_" + fn_str])
            with open(os.path.join(path, csv_file_name), 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(csv_data)

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='PASCALVOC2012_Segmentation\n\n' +
        'Download PASCAL VOC 2012 from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/.\n'
        'Note: 2 GB of data will be downloaded, and additionally 1 GB of data will be created. This process may take a long time. Please pay attention to the free space of the disk.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=PASCALVOC2012_Segmentation',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
