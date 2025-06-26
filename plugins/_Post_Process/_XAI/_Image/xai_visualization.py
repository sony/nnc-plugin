# Copyright 2023,2024,2025 Sony Group Corporation.
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
import os
import csv
from nnabla import logger
from nnabla.utils.image_utils import imsave
from gradcam_utils.gradcam_func import gradcam_func
from lime_utils.lime_func import lime_func
from shap_utils.shap_func import shap_func
from smoothgrad_utils.smoothgrad_func import smoothgrad_func


def func(args):
    if (not args.GradCAM) and (not args.LIME) and (not args.SHAP) and (not args.SmoothGrad):
        logger.critical('Task to run is not found.')
        raise RuntimeError(
            "At least one task (Grad-CAM, LIME, SHAP, SmoothGrad) is necessary to use this plugin.")

    data_output_dir = os.path.splitext(args.output)[0]

    class Args:
        pass

    header = ['Original']
    rows = {'Original': args.image}

    if args.GradCAM:
        gradcam_args = Args()
        gradcam_args.model = args.model
        gradcam_args.input = args.input
        gradcam_args.image = args.image
        gradcam_args.class_index = args.class_index
        gradcam_args.output_variable = args.output_variable
        gradcam_args.output = data_output_dir + '_' + 'gradcam.png'
        gradcam_args.contains_crop = args.contains_crop
        gradcam_result = gradcam_func(gradcam_args)
        imsave(gradcam_args.output, gradcam_result, channel_first=True)
        header.append('Grad-CAM')
        grad_cam_row = {'Grad-CAM': gradcam_args.output}
        rows.update(grad_cam_row)

    if args.SmoothGrad:
        smoothgrad_args = Args()
        smoothgrad_args.model = args.model
        smoothgrad_args.noise_level = args.noise_level
        smoothgrad_args.num_samples = args.num_samples_smoothgrad
        smoothgrad_args.layer_index = args.layer_index
        smoothgrad_args.image = args.image
        smoothgrad_args.class_index = args.class_index
        smoothgrad_args.output = data_output_dir + '_' + 'smoothgrad.png'
        smoothgrad_result = smoothgrad_func(smoothgrad_args)
        imsave(smoothgrad_args.output, smoothgrad_result, channel_first=True)
        header.append('SmoothGrad')
        smoothgrad_row = {'SmoothGrad': smoothgrad_args.output}
        rows.update(smoothgrad_row)

    if args.LIME:
        lime_args = Args()
        lime_args.model = args.model
        lime_args.image = args.image
        lime_args.class_index = args.class_index
        lime_args.num_samples = args.num_samples_lime
        lime_args.num_segments = args.num_segments
        lime_args.num_segments_2 = args.num_segments_2
        lime_args.output = data_output_dir + '_' + 'lime.png'
        lime_result = lime_func(lime_args)
        imsave(lime_args.output, lime_result)
        header.append('LIME')
        lime_row = {'LIME': lime_args.output}
        rows.update(lime_row)

    if args.SHAP:
        shap_args = Args()
        shap_args.image = args.image
        shap_args.input = args.input
        shap_args.model = args.model
        shap_args.class_index = args.class_index
        shap_args.num_samples = args.num_samples_shap
        shap_args.batch_size = args.batch_size
        shap_args.interim_layer = args.interim_layer
        shap_args.output = data_output_dir + '_' + 'shap.png'
        shap_func(shap_args)
        header.append('SHAP')
        shap_row = {'SHAP': shap_args.output}
        rows.update(shap_row)

    with open(args.output, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(rows)


def main():
    parser = argparse.ArgumentParser(
        description='XAI Visualization (GradCAM, SHAP, LIME, SmoothGrad)\n' +
        'Grad-CAM\n' +
        '\n' +
        'Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization\n' +
        'Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra\n' +
        'Proceedings of the IEEE International Conference on Computer Vision, 2017.\n' +
        'https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html\n' +
        '' +
        'LIME (image)\n' +
        '\n' +
        '"Why Should I Trust You?": Explaining the Predictions of Any Classifier\n' +
        'Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin\n' +
        'Knowledge Discovery and Data Mining, 2016.\n' +
        'https://dl.acm.org/doi/abs/10.1145/2939672.2939778\n' +
        '' +
        'SHAP\n'
        '\n'
        'A Unified Approach to Interpreting Model Predictions\n'
        'Scott Lundberg, Su-In Lee\n' +
        'Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017.\n' +
        'https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html\n' +
        '' +
        'SmoothGrad\n' +
        '\n' +
        'SmoothGrad: removing noise by adding noise\n' +
        'Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viegas, Martin Wattenberg\n' +
        'Workshop on Visualization for Deep Learning, ICML, 2017.\n' +
        'https://arxiv.org/abs/1706.03825\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-m', '--model', help='path to model nnp file (model) default=results.nnp', required=True, default='results.nnp')
    parser.add_argument(
        '-i', '--image', help='path to input image file (image)', required=True)
    parser.add_argument(
        '-c', '--class_index', help='class index to visualizeil (int), default=0', required=True, type=int, default=0)
    parser.add_argument(
        '-o', '--output', help='path to output image file (image) default=xai_visualization.csv', required=True, default='xai_visualization.csv')
    # designage if the model contains crop between input and first conv layer.
    parser.add_argument(
        '-cr', '--contains_crop', help=argparse.SUPPRESS, action='store_true')
    parser.add_argument(
        '-ov', '--output_variable', help="(Grad-CAM) output variable to visualize (variable) default=y0' ", required=True, default="y0'")
    parser.add_argument(
        '-s', '--num_segments', help='(LIME) number of segments (int), default=10', required=True, type=int, default=10)
    parser.add_argument(
        '-s2', '--num_segments_2', help='(LIME) number of segments to highlight (int), default=3', required=True, type=int, default=3)
    parser.add_argument(
        '-n_lime', '--num_samples_lime', help='(LIME) number of samples N (int), default=1000', required=True, type=int, default=1000)
    parser.add_argument(
        '-n_shap', '--num_samples_shap', help='(SHAP) number of samples N (int), default=100', required=True, type=int, default=100)
    parser.add_argument(
        '-n_smoothgrad', '--num_samples_smoothgrad', help='(SmoothGrad) number of samples N (int), default=25', required=True, type=int, default=25)
    parser.add_argument(
        '-in', '--input', help='path to input csv file (csv) default=output_result.csv', required=True, default='output_result.csv')
    parser.add_argument(
        '-b', '--batch_size', help='(SHAP) batch size, default=50', required=True, type=int, default=50)
    # index of layer of interest to visualize (input layer is 0), default=0
    parser.add_argument(
        '-il', '--interim_layer', help='(SHAP) layer input to explain, default=0', required=True, type=int, default=0)
    parser.add_argument(
        '-nl', '--noise_level', help='noise level(0.0 to 1.0) to calculate standard deviation for input image, default=0.15', type=float, default=0.15)
    parser.add_argument(
        '-li', '--layer_index', help=argparse.SUPPRESS, type=int, default=0)
    parser.add_argument(
        '-g', '--GradCAM', action='store_true', help='Grad-CAM (bool), default=True')
    parser.add_argument(
        '-sm', '--SmoothGrad', action='store_true', help='SmoothGrad (bool), default=True')
    parser.add_argument(
        '-l', '--LIME', action='store_true', help='LIME(image) (bool), default=True')
    parser.add_argument(
        '-sh', '--SHAP', action='store_true', help='SHAP (bool), default=True')

    args = parser.parse_args()
    func(args)
    logger.log(
        99, 'XAI Visualization (GradCAM, SHAP, LIME, SmoothGrad) (image) completed successfully.')


if __name__ == '__main__':
    main()
