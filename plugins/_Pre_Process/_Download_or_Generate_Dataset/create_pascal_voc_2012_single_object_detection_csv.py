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
import xml.etree.ElementTree as ET
from nnabla.utils.image_utils import imresize, imsave
from PIL import Image, ImageDraw

from nnabla.logger import logger
from nnabla.utils.data_source_loader import download


def enum_img_ids(tar_file, file_name):
    file_info = tar_file.getmember(file_name)
    f = tar_file.extractfile(file_info)
    img_ids = [id.decode('utf-8').strip() for id in f.readlines()]
    f.close()
    return img_ids


def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def pad_image(im):
    h = im.shape[0]
    w = im.shape[1]

    pad = (((500 - h) // 2, 500 - h - (500 - h) // 2),
           ((500 - w) // 2, 500 - w - (500 - w) // 2))
    if len(im.shape) == 3:
        pad = pad + ((0, 0),)
    im = np.pad(im, pad, 'constant', constant_values=0)
    assert(im.shape[0] == 500 and im.shape[1] == 500)

    return im, pad


def convert_image_and_create_csv(tar_file, img_ids, resolution, target_dir, csv_file_name):
    # CSV header
    csv_data = [['x:image', 'y:catdog', 'r:region']]

    # Create output dir
    makedirs(os.path.join(target_dir, 'images_' + resolution))
    makedirs(os.path.join(target_dir, 'labels_' + resolution))

    # Convert images
    def parse_xml(img_id):
        cls_ids = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                   "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        annotation_info = tar_file.getmember(
            "VOCdevkit/VOC2012/Annotations/" + img_id + ".xml")
        with tar_file.extractfile(annotation_info) as f:
            xml = f.read()
        root = ET.fromstring(xml)
        objects = root.findall('object')
        found = False
        for object in objects:
            cls = object.find('name').text
            if cls_ids.index(cls) == 7 or cls_ids.index(cls) == 11:
                if found:
                    # more than 2 objects
                    return -1, 0.5, 0.5, 1.0, 1.0
                # image with single object
                found = True
                bndbox = object.find('bndbox')
                xmin, ymin, xmax, ymax = float(bndbox.find('xmin').text), float(bndbox.find(
                    'ymin').text), float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)
        if found:
            return 1, (xmax + xmin) * 0.5, (ymax + ymin) * 0.5, xmax - xmin, ymax - ymin
        else:
            return 0, 0.5, 0.5, 1.0, 1.0

    for img_id in tqdm.tqdm(img_ids):
        class_id, x, y, w, h = parse_xml(img_id)
        if class_id >= 0:
            # open source image
            try:
                img_file_info = tar_file.getmember(
                    "VOCdevkit/VOC2012/JPEGImages/" + img_id + ".jpg")
                with tar_file.extractfile(img_file_info) as f:
                    img = np.asarray(Image.open(f))
                assert(len(img.shape) == 3)
                assert(img.shape[2] == 3)
                assert(img.shape[0] <= 500 and img.shape[1] <= 500)

                # Image
                img, pad = pad_image(img)
                if class_id > 0:
                    x, y = x + pad[1][0], y + pad[0][0]
                if resolution == "125px":
                    img = imresize(img, size=(125, 125))
                    x, y, w, h = x/4, y/4, w/4, h/4
                imsave(os.path.join(target_dir, 'images_' +
                       resolution, img_id + '.png'), img)

                # Label
                label = Image.new('L', (img.shape[0], img.shape[1]), 0)
                draw = ImageDraw.Draw(label)
                if class_id:
                    draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], 255)
                label.save(os.path.join(target_dir, 'labels_' +
                           resolution, img_id + '.png'))

                csv_data.append(["./images_" + resolution + "/" + img_id + ".png",
                                str(class_id), "./labels_" + resolution + "/" + img_id + ".png"])
            except:
                logger.warning(
                    "Failed to convert %s." % (img_id))
                raise

    # Save CSV
    with open(os.path.join(target_dir, csv_file_name), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(csv_data)


def func(args):
    path = args.output_dir

    # Download PASCAL VOC 2012 dataset
    logger.log(99, 'Downloading PASCAL VOC 2012 dataset...')
    dataset_file = download(
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")

    # Open tar file
    tar_file = tarfile.open(fileobj=dataset_file)

    # Enumerate images　for segmentation
    train_img_ids = enum_img_ids(
        tar_file, "VOCdevkit/VOC2012/ImageSets/Main/train.txt")
    val_img_ids = enum_img_ids(
        tar_file, "VOCdevkit/VOC2012/ImageSets/Main/val.txt")

    # Convert training images
    for resolution in ["500px", "125px"]:
        logger.log(99, 'Converting training images ... ' + resolution)
        convert_image_and_create_csv(tar_file, train_img_ids, resolution, path,
                                     "pascal_voc_2012_single_object_detection_train_" + resolution + ".csv")

        logger.log(99, 'Converting validation images ...' + resolution)
        convert_image_and_create_csv(tar_file, val_img_ids, resolution, path,
                                     "pascal_voc_2012_single_object_detection_val_" + resolution + ".csv")

    logger.log(99, 'Dataset creation completed successfully.')


def main():
    parser = argparse.ArgumentParser(
        description='PASCALVOC2012_SingleObjectDetection\n\n' +
        'Download PASCAL VOC 2012 from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/.\n'
        'Note: 2 GB of data will be downloaded, and additionally 1 GB of data will be created. This process may take a long time. Please pay attention to the free space of the disk.\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output-dir',
        help='path to write NNC dataset CSV format (dir) default=PASCALVOC2012_SingleObjectDetection',
        required=True)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
