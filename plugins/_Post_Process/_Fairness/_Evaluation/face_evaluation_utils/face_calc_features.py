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
import numpy as np
import cv2
import os
import csv


# save to csv file
def save_to_csv(filename, header, dict_to_save, data_type):
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(np.array(
            [(key, value) for key, value in dict_to_save.items()], dtype=data_type))


# mask the image for face
def make_masked_face_image(img, face, w_f=0.2, w_l=0.8, h_f=0, h_l=1, blocks=3):
    x, y, h, w = face[0], face[1], face[2], face[3]
    faces = []
    for i in range(blocks):
        for j in range(blocks):
            mask = np.zeros_like(img)
            w_1 = int((w_f + (i / blocks) * (w_l - w_f)) * w)
            w_2 = int((w_f + ((i + 1) / blocks) * (w_l - w_f)) * w)
            h_1 = int((w_f + (j / blocks) * (h_l - h_f)) * h)
            h_2 = int((w_f + ((j + 1) / blocks) * (h_l - h_f)) * h)
            pts = np.array(((x + w_1, y + h_1), (x + w_2, y + h_1),
                            (x + w_2, y + h_2), (x + w_1, y + h_2)))
            cv2.fillConvexPoly(mask, pts, (255, 255, 255))
            face = np.where(mask == 255, img, mask)
            faces.append(face)
    return faces


def extract_non_black(img):
    img_total = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    non_black = np.where(img_total > 0)
    extract_img = []
    for i in range(len(non_black[0])):
        extract_img.append(img[non_black[0][i], non_black[1][i], :])
    extract_img = np.array(extract_img).reshape(1, -1, 3)
    return extract_img


# calculate ITA
def calc_ita(masked_imgs):
    itas = []
    for masked_img in masked_imgs:
        lab_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2Lab)
        l_img, a_img, b_img = cv2.split(lab_img)
        l_standardize_before = 255
        l_standardize_after = 100
        b_standardize = 128
        white_point = 50
        rad_to_deg = 180 / np.pi
        bins = 1000

        l_img = l_img * (l_standardize_after / l_standardize_before)
        b_img = b_img - b_standardize
        ITA_img = (np.arctan((l_img - white_point) / b_img)) * rad_to_deg
        a_hist, a_bins = np.histogram(ITA_img, bins=bins)
        ITA_each = a_bins[a_hist.argmax()]
        itas.append(ITA_each)
    return round(sum(itas) / len(itas), 2)


def make_ita_dict(input_csv, output):
    model_name = "haarcascade_frontalface_alt2.xml"
    dir_path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(dir_path, model_name)
    cas = cv2.CascadeClassifier(model_path)
    paths = []
    with open(input_csv) as f:
        for line in f:
            paths.append(line.split(",")[0].rstrip())
    results = {}
    csv_path = os.path.abspath(os.path.dirname(input_csv))
    for path in paths[1:]:
        absolute_path = os.path.join(csv_path, path[2:])
        img = cv2.imread(absolute_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cas.detectMultiScale(
            gray_img, scaleFactor=1.25, minNeighbors=2)
        if len(faces) == 0:
            results[absolute_path] = 'No face detected'
            continue
        if len(faces) >= 2:
            results[absolute_path] = 'More than two faces detected'
            continue
        else:
            masked_img = make_masked_face_image(img, faces[0])
            masked_img = [extract_non_black(img) for img in masked_img]
            ita_score = calc_ita(masked_img)
            results[absolute_path] = ita_score
    save_to_csv(filename=output, header=[
                'x:image', 'ITA'], dict_to_save=results, data_type='object')
