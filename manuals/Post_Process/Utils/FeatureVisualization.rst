Utils/Feature Vislization
~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize intermediate activation of the model with a single piece of data.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Specify the trained model file (*.nnp) to use for vizualization.
        
        To perform vizualization based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - input-variable
     - Specify the variable name of the trained model to input the data specified by input-data for vizualization.

   * - input-data
     -
        Specify the input data to use for vizualization.
        
        To perform vizualization on specific data shown in the evaluation results of the Evaluation tab, start the plugin with the cell containing the image file name selected. The data file name will be automatically input in input-data.

   * - layer-name
     - Specify the name of the layer to visualize the output.

   * - output
     - Specify the name of the CSV file to output the vizualization results to.

Utils/Feature Vislization (batch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize intermediate activation of the model for all the data contained in the dataset.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     - Specify the dataset CSV file containing the data to use for visualization.

   * - model
     -
        Specify the trained model file (*.nnp) to use for vizualization.
        
        To perform vizualization based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - layer-name
     - Specify the name of the layer to visualize the output.

   * - output
     - Specify the name of the CSV file to output the vizualization results to.

