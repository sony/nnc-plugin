XAI/LIME(image)
~~~~~~~~~~~~~~~

Using a method called LIME, the areas of the input image that affect the classification result are made visible in the model, which performs image classification.

`Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Specify the model file (*.nnp) that will be used in the LIME computation.
        
        To perform LIME based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - image
     -
        Specify the image file to analyze.
        
        To perform LIME on a specific image shown in the evaluation results of the Evaluation tab, start the plugin with the cell containing the image file name selected. The image file name will be automatically input in image.

   * - class_index
     -
        Specify the index of the class to perform visualization on.
        
        By default, visualization is performed on class number 0.

   * - num_samples
     - Specify the number of times to sample the relationship between input image and classification result.

   * - num_segments
     - Specify the number of segments to divide the input image.

   * - num_segments_2
     - Specify the number of segments to make visible in the area divided into num_segments.

   * - output
     -
        Specify the name of the image file to output the visualization results to.
        
        If LIME is executed from the evaluation results of the Evaluation tab, the visualization results are saved to the specified file in the training result folder.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as PNG file.


XAI/LIME(image batch)
~~~~~~~~~~~~~~~~~~~~~

Using a method called LIME, the areas of the input image that affect the classification result are made visible in the model, which performs image classification. LIME(batch) processes all images in the specified dataset, while LIME processes a single image. This plugin can be used both for binary of multi class classifications.

`Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     - Specify the dataset CSV file containing image files to be processed by LIME. To process LIME on images in the Output Result shown in the Evaluation tab, use the default output_result.csv.

   * - model
     -
        Specify the model file (*.nnp) that will be used in the LIME computation.
        
        To perform LIME based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - input_variable
     - Of the variables included in the dataset CSV file specified in input, specify the name of the variable to be used in the LIME computation.

   * - label_variable
     - Of the variables included in the dataset CSV file specified in input, specify the variable name of the class index to be visualized.

   * - num_samples
     - Specify the number of times to sample the relationship between input image and classification result.

   * - num_segments
     - Specify the number of segments to divide the input image.

   * - num_segments_2
     - Specify the number of segments to make visible in the area divided into num_segments.

   * - output
     -
        Specify the dataset CSV file to output the visualization results to.
        
        If LIME(batch) is executed from the evaluation results of the Evaluation tab, the visualization results are saved to the specified file in the training result folder.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.
The other columns than listed below are the same meaning as those in output_result.csv file that is generated as a result of evaluation.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - lime
     - The result path of this plugin for the target instance, the image of which is displayed in NNC window.

XAI/LIME(tabular)
~~~~~~~~~~~~~~~~~

Using a method called LIME, a classification result is explained with the contribution of the features in input table data. Each feature is explained with a set of inequality and degree of contribution, which enables to interpret the classifier judgement. This plugin supports regression model and classification model with categorical features as well as model with continuous values.

`Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Specify the model file (*.nnp) that will be used in the LIME computation.
        
        To perform LIME based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - input
     - Specify the dataset CSV file containing the data to analyze.

   * - categorical
     - Specify the indices of the columns in the input CSV where categorical features are used. It has to be given in integers separated with comma.

   * - index
     - Specify the index of the data in the input CSV.

   * - class_index
     - Specify the index of the class of the data to analyze. Default value is 0. For regression model or binary classification model, only class_index=0 can be specified.

   * - num_samples
     - Specify the number of times to sample the relationship between input data and classification result.

   * - train
     - Specify the dataset CSV file used for the training of the model of interest.

   * - output
     - Specify the name of the CSV file to output the processing results to.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the rows and columns of CSV file is as follows.
The 'Sample (Index {n})' row represents the value of each feature, the name of which corresponds to each column name in output_result.csv.
The 'Importance' row shows the importance of each input feature in the classification. The row above 'Importance' means the feature range that gives the importance.

XAI/LIME(tabular batch)
~~~~~~~~~~~~~~~~~~~~~~~

Using a method called LIME, a classification result is explained with the contribution of the features in input table data. Each feature is explained with a set of inequality and degree of contribution, which enables to interpret the classifier judgement. This plugin supports regression model and classification model with categorical features as well as model with continuous values. LIME(tabular batch) processes all records in the specified dataset, while LIME(tabular) processes a single record.

`Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?: Explaining the Predictions of Any Classifier". Knowledge Discovery and Data Mining, 2016. <https://dl.acm.org/doi/abs/10.1145/2939672.2939778>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Specify the model file (*.nnp) that will be used in the LIME computation.
        
        To perform LIME based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - input
     - Specify the dataset CSV file containing the data to analyze.

   * - categorical
     - Specify the indices of the columns in the input CSV where categorical features are used. It has to be given in integers separated with comma.

   * - class_index
     - Specify the index of the class of the data to analyze. Default value is 0. For regression model or binary classification model, only class_index=0 can be specified.

   * - num_samples
     - Specify the number of times to sample the relationship between input data and classification result.

   * - train
     - Specify the dataset CSV file used for the training of the model of interest.

   * - output
     - Specify the name of the CSV file to output the processing results to.

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
