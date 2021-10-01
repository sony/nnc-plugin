Utils/Inference
~~~~~~~~~~~~~~~

Inference is performed on new data using a trained model.



.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Specify the trained model file (*.nnp) to use for inference.
        
        To perform inference based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - input-variable
     - Specify the variable name of the trained model to input the data specified by input-data for inference.

   * - input-data
     -
        Specify the input data to use for inference.
        
        To perform inference on specific data shown in the evaluation results of the Evaluation tab, start the plugin with the cell containing the image file name selected. The data file name will be automatically input in input-data.

   * - output
     - Specify the name of the CSV file to output the inference results to.


