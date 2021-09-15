XAI/SmoothGrad
~~~~~~~~~~~~~~

Using a method called SmoothGrad, the areas of the input image that affect the classification result are made visible in the model, which performs image classification.

SmoothGrad: removing noise by adding noise
   - Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg
   - https://arxiv.org/abs/1706.03825

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     - Specify the model file (*.nnp) that will be used in the SmoothGrad computation. To perform SmoothGrad based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - noise_level
     - Specify the noise level (0.0 to 1.0) to calculate standard deviation used for generating gausian noise.

   * - num_samples
     - Specify the number of times of gradient calculation to get the final averaged sensitivity map. Gausian noise is repeatedly generated and put onto the input image to calculate gradient of each time.

   * - image
     - Specify the image file to analyze. To perform SmoothGrad on a specific image shown in the evaluation results of the Evaluation tab, start the plugin with the cell containing the image file name selected. The image file name will be automatically input in image.

   * - class_index
     - Specify the index of the class to perform visualization on. By default, visualization is performed on class number 0.

   * - output
     - Specify the name of the image file to output the visualization results to. If SmoothGrad is executed from the evaluation results of the Evaluation tab, the visualization results are saved to the specified file in the training result folder.


XAI/SmoothGrad (batch)
~~~~~~~~~~~~~~~~~~~~~~

Using a method called SmoothGrad, the areas of the input image that affect the classification result are made visible in the model, which performs image classification.

SmoothGrad: removing noise by adding noise
   - Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg
   - https://arxiv.org/abs/1706.03825

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     - Specify the model file (*.nnp) that will be used in the SmoothGrad computation. To perform SmoothGrad based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - noise_level
     - Specify the noise level (0.0 to 1.0) to calculate standard deviation used for generating gausian noise.

   * - num_samples
     - Specify the number of times of gradient calculation to get the final averaged sensitivity map. Gausian noise is repeatedly generated and put onto the input image to calculate gradient of each time.

   * - input
     - Specify the dataset CSV file containing image files to be processed by SmoothGrad. To process SmoothGrad on images in the Output Result shown in the Evaluation tab, use the default output_result.csv.

   * - output
     - Specify the name of the image file to output the visualization results to. If SmoothGrad is executed from the evaluation results of the Evaluation tab, the visualization results are saved to the specified file in the training result folder.

   * - input_variable
     - Of the variables included in the dataset CSV file specified in input, specify the name of the variable to be used in the SmoothGrad computation.

   * - label_variable
     - Of the variables included in the dataset CSV file specified in input, specify the variable name of the class index to be visualized.
