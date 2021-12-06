XAI/Grad-CAM
~~~~~~~~~~~~

Using a method called Grad-CAM, the areas of the input image that affect the classification result are made visible in Convolutional Neural Networks, which performs image classification.
At least one convolution layer is necessary in the model to use this plugin.

`Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". Proceedings of the IEEE International Conference on Computer Vision, 2017. <https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Specify the Convolutional Neural Networks model file (*.nnp) that will be used in the Grad-CAM computation.
        
        To perform Grad-CAM based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - image
     -
        Specify the image file to analyze.
        
        To perform Grad-CAM on a specific image shown in the evaluation results of the Evaluation tab, start the plugin with the cell containing the image file name selected. The image file name will be automatically input in image.

   * - class_index
     -
        Specify the index of the class to perform visualization on.
        
        By default, visualization is performed on class number 0.

   * - output
     -
        Specify the name of the image file to output the visualization results to.
        
        If Grad-CAM is executed from the evaluation results of the Evaluation tab, the visualization results are saved to the specified file in the training result folder.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as PNG file.
It is shown in jet colormap over the original image, where reddish color means the positively affected area.

XAI/Grad-CAM(batch)
~~~~~~~~~~~~~~~~~~~

Using a method called Grad-CAM, the areas of the input image that affect the classification result are made visible in Convolutional Neural Networks, which performs image classification. Grad-CAM(batch) processes all images in the specified dataset, while Grad-CAM processes a single image.
At least one convolution layer is necessary in the model to use this plugin.

`Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". Proceedings of the IEEE International Conference on Computer Vision, 2017. <https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     -
        Specify the dataset CSV file containing image files to be processed by Grad-CAM.
        
        To process Grad-CAM on images in the Output Result shown in the Evaluation tab, use the default output_result.csv.

   * - model
     -
        Specify the Convolutional Neural Networks model file (*.nnp) that will be used in the Grad-CAM computation.
        
        To perform Grad-CAM based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - input_variable
     - Of the variables included in the dataset CSV file specified in input, specify the name of the variable to be used in the Grad-CAM computation.

   * - label_variable
     - Of the variables included in the dataset CSV file specified in input, specify the variable name of the class index to be visualized.

   * - output
     -
        Specify the dataset CSV file to output the visualization results to.
        
        If Grad-CAM is executed from the evaluation results of the Evaluation tab, the visualization results are saved to the specified file in the training result folder.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.
The other columns than listed below are the same meaning as those in output_result.csv file that is generated as a result of evaluation.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - gradcam
     - The result path of this plugin for the target instance, the image of which is displayed in NNC window. It is shown in jet colormap over the original image, where reddish color means the positively affected area.
