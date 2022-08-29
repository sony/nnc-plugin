XAI/XAI Visualization
~~~~~~~~~~~~~~~~~~~~~

Using four methods called Grad-CAM,LIME,SHAP and SmoothGrad, the areas of the input image that affect the classification result are made visible in the model, which performs image classification.

| Grad-CAM:
| `Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". Proceedings of the IEEE International Conference on Computer Vision, 2017. <https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>`_
|
| LIME:
| `Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_
|
| SHAP:
| `Scott M Lundberg, Su-In Lee. "A unified approach to interpreting model predictions". Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017. <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_
|
| SmoothGrad:
| `Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg. "Smoothgrad: removing noise by adding noise". Workshop on Visualization for Deep Learning, ICML, 2017. <https://arxiv.org/abs/1706.03825>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Specify the model file (*.nnp) that will be used in the XAI Visualization computation.
        
        To perform XAI Visualization based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - image
     -
        Specify the image file to analyze.
        
        To perform XAI Visualization on a specific image shown in the evaluation results of the Evaluation tab, start the plugin with the cell containing the image file name selected. The image file name will be automatically input in image.

   * - class_index
     -
        Specify the index of the class to perform visualization on.
        
        By default, visualization is performed on class number 0.

   * - output
     -
        Specify the name of the image file to output the visualization results to.
        
        If XAI Visualization is executed from the evaluation results of the Evaluation tab, the visualization results are saved to the specified file in the training result folder.

   * - num_segments
     - For LIME processing, specify the number of segments to divide the input image.

   * - num_segments_2
     - For LIME processing, specify the number of segments to make visible in the area divided into num_segments.

   * - num_samples_lime
     - For LIME processing, specify the number of times to sample the relationship between input image and classification result.

   * - num_samples_shap
     - For SHAP processing, specify the number of times to sample the relationship between input image and classification result.

   * - num_samples_smoothgrad
     - For SmoothGrad processing, specify the number of times to sample the relationship between input image and classification result.

   * - input
     - For SHAP processing, specify the dataset CSV file containing image files to be processed. To process SHAP on images in the Output Result shown in the Evaluation tab, use the default output_result.csv.

   * - batch_size
     - For SHAP processing, specify the batch size to process with.

   * - interim_layer
     - For SHAP processing, specify the layer of interest for which SHAP computation is executed. Designate the layer counts from the input layer (input is counted as 0th layer). By default, it is processed with input layer.

   * - noise_level
     - For SmoothGrad processing, specify the noise level (0.0 to 1.0) to calculate standard deviation used for generating gausian noise.

   * - GradCAM
     - Execute Grad-CAM if designated (By default, not executed).

   * - SmoothGrad
     - Execute SmoothGrad if designated (By default, not executed).

   * - LIME
     - Execute LIME if designated (By default, not executed).

   * - SHAP
     - Execute SHAP if designated (By default, not executed).

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.
The other columns than listed below are the same meaning as those in output_result.csv file that is generated as a result of evaluation.
None of the 4 tasks (Grad-CAM, SmoothGrad, LIME, SHAP) are executed by default when called from command line, while Grad-CAM and SmoothGrad are executed by default when called from Neural Network Console.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - gradcam
     - The result path of Grad-CAM for the target instance, the image of which is displayed in NNC window. It is shown in jet colormap over the original image, where reddish color means the positively affected area.

   * - lime
     - The result path of LIME for the target instance, the image of which is displayed in NNC window.

   * - shap
     - The result path of SHAP for the target instance, the image of which is shown in NNC window. It is shown in red over the original image for the positively affected area in the classification, while in blue for the negatively affected area.

   * - SmoothGrad
     - The result path of SmoothGrad for the target instance, the image of which is displayed in NNC window. It is shown in grayscale image as sensitivity map.


XAI/XAI Visualization(batch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using four methods called Grad-CAM,LIME,SHAP and SmoothGrad, the areas of the input image that affect the classification result are made visible in Convolutional Neural Networks, which performs image classification. XAI Visualization(batch) processes all images in the specified dataset, while XAI Visualization processes a single image.


| Grad-CAM:
| `Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". Proceedings of the IEEE International Conference on Computer Vision, 2017. <https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>`_
|
| LIME:
| `Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_
|
| SHAP:
| `Scott M Lundberg, Su-In Lee. "A unified approach to interpreting model predictions". Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017. <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_
|
| SmoothGrad:
| `Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg. "Smoothgrad: removing noise by adding noise". Workshop on Visualization for Deep Learning, ICML, 2017. <https://arxiv.org/abs/1706.03825>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     -
        Specify the dataset CSV file containing image files to be processed by XAI Visualization.
        
        To process XAI Visualization on images in the Output Result shown in the Evaluation tab, use the default output_result.csv.

   * - model
     -
        Specify the Convolutional Neural Networks model file (*.nnp) that will be used in the XAI Visualization computation.
        
        To perform XAI Visualization based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - input_variable
     - Of the variables included in the dataset CSV file specified in input, specify the name of the variable to be used in the XAI Visualization computation.

   * - label_variable
     - Of the variables included in the dataset CSV file specified in input, specify the variable name of the class index to be visualized.

   * - output
     -
        Specify the dataset CSV file to output the visualization results to.
        
        If XAI Visualization is executed from the evaluation results of the Evaluation tab, the visualization results are saved to the specified file in the training result folder.

   * - num_segments
     - For LIME processing, specify the number of segments to divide the input image.

   * - num_segments_2
     - For LIME processing, specify the number of segments to make visible in the area divided into num_segments.

   * - num_samples_lime
     - For LIME processing, specify the number of times to sample the relationship between input image and classification result.

   * - num_samples_shap
     - For SHAP processing, specify the number of times to sample the relationship between input image and classification result.

   * - num_samples_smoothgrad
     - For SmoothGrad processing, specify the number of times to sample the relationship between input image and classification result.

   * - input
     - For SHAP processing, specify the dataset CSV file containing image files to be processed. To process SHAP on images in the Output Result shown in the Evaluation tab, use the default output_result.csv.

   * - batch_size
     - For SHAP processing, specify the batch size to process with.

   * - interim_layer
     - For SHAP processing, specify the layer of interest for which SHAP computation is executed. Designate the layer counts from the input layer (input is counted as 0th layer). By default, it is processed with input layer.

   * - noise_level
     - For SmoothGrad processing, specify the noise level (0.0 to 1.0) to calculate standard deviation used for generating gausian noise.

   * - GradCAM
     - Execute Grad-CAM if designated (By default, not executed).

   * - SmoothGrad
     - Execute SmoothGrad if designated (By default, not executed).

   * - LIME
     - Execute LIME if designated (By default, not executed).

   * - SHAP
     - Execute SHAP if designated (By default, not executed).

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.
The other columns than listed below are the same meaning as those in output_result.csv file that is generated as a result of evaluation.
None of the 4 tasks (Grad-CAM, SmoothGrad, LIME, SHAP) are executed by default when called from command line, while Grad-CAM and SmoothGrad are executed by default when called from Neural Network Console.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - gradcam
     - The result path of Grad-CAM for the target instance, the image of which is displayed in NNC window. It is shown in jet colormap over the original image, where reddish color means the positively affected area.

   * - lime
     - The result path of LIME for the target instance, the image of which is displayed in NNC window.

   * - shap
     - The result path of SHAP for the target instance, the image of which is shown in NNC window. It is shown in red over the original image for the positively affected area in the classification, while in blue for the negatively affected area.

   * - SmoothGrad
     - The result path of SmoothGrad for the target instance, the image of which is displayed in NNC window. It is shown in grayscale image as sensitivity map.