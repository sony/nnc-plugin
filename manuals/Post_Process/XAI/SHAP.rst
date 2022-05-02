XAI/SHAP(Image)
~~~~~~~~~~~~~~~

Using a method called SHAP, the areas of the input image that affect the classification result are made visible in the model, which performs image classification.

`Scott M Lundberg, Su-In Lee. "A unified approach to interpreting model predictions". Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017. <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - image
     - Specify the image file to analyze. To perform SHAP on a specific image shown in the evaluation results of the Evaluation tab, start the plugin with the cell containing the image file name selected. The image file name will be automatically input in image.

   * - input
     - Specify the dataset CSV file containing image files to be processed by SHAP. To process SHAP on images in the Output Result shown in the Evaluation tab, use the default output_result.csv.

   * - model
     - Specify the model file (*.nnp) that will be used in the SHAP computation. To perform SHAP based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - class_index
     - Specify the index of the class to perform visualization on. By default, visualization is performed on class number 0.

   * - num_samples
     - Specify the number of times to sample the relationship between input image and classification result.

   * - batch_size
     - Specify the batch size to process with SHAP.

   * - interim_layer
     - Specify the layer of interest for which SHAP computation is executed. Designate the layer counts from the input layer (input is counted as 0th layer). By default, it is processed with input layer.

   * - output
     - Specify the name of the image file to output the visualization results to.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as PNG file.
It is shown in red over the original image for the positively affected area in the classification, while in blue for the negatively affected area.

XAI/SHAP(Image batch)
~~~~~~~~~~~~~~~~~~~~~

Using a method called SHAP, the areas of the input image that affect the classification result are made visible in the model, which performs image classification. SHAP(batch) processes all images in the specified dataset, while SHAP processes a single image.

`Scott M Lundberg, Su-In Lee. "A unified approach to interpreting model predictions". Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017. <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Specify the dataset CSV file containing image files to be processed by SHAP. To process SHAP on images in the Output Result shown in the Evaluation tab, use the default output_result.csv.

   * - model
     - Specify the model file (*.nnp) that will be used in the SHAP computation. To perform SHAP based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - input_variable
     - Of the variables included in the dataset CSV file specified in input, specify the name of the variable to be used in the SHAP computation.

   * - label_variable
     - Of the variables included in the dataset CSV file specified in input, specify the variable name of the class index to be visualized.

   * - num_samples
     - Specify the number of times to sample the relationship between input image and classification result.

   * - batch_size
     - Specify the batch size to process with SHAP.

   * - interim_layer
     - Specify the layer of interest for which SHAP computation is executed. Designate the layer counts from the input layer (input is counted as 0th layer). By default, it is processed with input layer.

   * - Output
     - Specify the name of the CSV file to output the inference results to.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.
The other columns than listed below are the same meaning as those in output_result.csv file that is generated as a result of evaluation.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - shap
     - The result path of this plugin for the target instance, the image of which is shown in NNC window. It is shown in red over the original image for the positively affected area in the classification, while in blue for the negatively affected area.


XAI/Kernel SHAP (Tabular)
~~~~~~~~~~~~~~~~~~~~~~~~~

Using a method called Kernel SHAP, a classification result is
explained with the contribution of the features in input table
data. Each feature is explained with degree of contribution, which
enables to interpret the classifier judgement.

`Scott M Lundberg, Su-In Lee. "A unified approach to interpreting model predictions". Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017. <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - model
     - Specify the model file (*.nnp) that will be used in the Kernel SHAP computation. To perform Kernel SHAP based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - input
     - Specify the dataset CSV file containing the data to analyze.

   * - train
     - Specify the dataset CSV file used for the training of the model of interest.

   * - index
     - Specify the index of the data in the input CSV.

   * - alpha
     - Specify the coefficient for the regularization term of Ridge regression.

   * - class_index
     - Specify the index of the class of the data to analyze. Default value is 0. For regression model or binary classification model, only class_index=0 can be specified.

   * - output
     - Specify the name of the CSV file to output the inference results to.

   * - memory_limit
     - Specify the limit of memory based on which the process is stopped before memory error happens after a long period during plugin execution. The process stops when dataset size is expected to exceed the limit. By default, the limit value is speculated by calculation.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the rows and columns of CSV file is as follows.
The 'Sample (Index {n})' row represents the value of each feature, the name of which corresponds to each column name in output_result.csv.
The 'Importance' row shows the importance of each input feature in the classification.
The 'Image' column represents the visual summarization of Kernel SHAP, which is shown with the image and its path.


XAI/Kernel SHAP (Tabular Batch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using a method called Kernel SHAP, a classification result is
explained with the contribution of the features in input table
data. Each feature is explained with degree of contribution, which
enables to interpret the classifier judgement. Kernel SHAP(tabular
batch) processes all records in the specified dataset, while Kernel
SHAP(tabular) processes a single record.

`Scott M Lundberg, Su-In Lee. "A unified approach to interpreting model predictions". Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017. <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - model
     - Specify the model file (*.nnp) that will be used in the Kernel SHAP computation. To perform Kernel SHAP based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - input
     - Specify the dataset CSV file containing the data to analyze.

   * - train
     - Specify the dataset CSV file used for the training of the model of interest.

   * - class_index
     - Specify the index of the class of the data to analyze. Default value is 0. For regression model or binary classification model, only class_index=0 can be specified.

   * - alpha
     - Specify the coefficient for the regularization term of Ridge regression.

   * - output
     - Specify the name of the CSV file to output the inference results to.

   * - memory_limit
     - Specify the limit of memory based on which the process is stopped before memory error happens after a long period during plugin execution. The process stops when dataset size is expected to exceed the limit. By default, the limit value is speculated by calculation.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.
For the other columns, the column name of each feature represents the importance of target instance.


.. list-table::
   :widths: 30 70
   :class: longtable

   * - index
     - The index of the target instance in input-train dataset CSV file.
