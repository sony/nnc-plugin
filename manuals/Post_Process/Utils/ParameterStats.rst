Utils/Parameter Stats
~~~~~~~~~~~~~~~~~~~~~

This plugin calculates various statistics (size, maximum value, minimum value, absolute maximum value, absolute minimum value, absolute value, average, standard deviation) of the parameters included in the trained model.



.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Specify the model file (*.nnp) to calculate statistics.
        
        To calculate statistics based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - output
     -
        Specify the name of the CSV file to output the statistics to.
        
        If Parameter Stats is executed from the evaluation results of the Evaluation tab, a table summarizing the statistics is saved with the specified file name in the training result folder.


