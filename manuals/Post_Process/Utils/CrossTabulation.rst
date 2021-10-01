Utils/Cross Tabulation
~~~~~~~~~~~~~~~~~~~~~~

This plugin performs a cross tabulation on a dataset CSV file. It can be used to tabulate the number of data samples per label, calculate accuracies, and the like.



.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     -
        Specify the dataset CSV file to be processed.
        
        To perform cross tabulation on a CSV file containing evaluation results from Output Result of the Evaluation tab, use the default output_result.csv file.

   * - variable1
     - Specify the variable name to use for the rows of the cross tabulation result table.

   * - variable2
     - Specify the variable name to use for the columns of the cross tabulation result table.

   * - variable2_eval
     -
        To assign the results (correct/incorrect) of an accuracy evaluation performed on a variable specified by Variable2 to the columns of the cross tabulation result table, specify the variable name that will be used to compare to variable2.
        
        For example, to assign the results of comparing the correct label in image classification (y) and the results estimated by the neural network (y’) to the columns of the cross tabulation result table, set variable2 to “y” and variable2_eval to “y’.”
        
        If you specify blank, the value specified by varable2 is assigned to the column of the cross tabulation result table.

   * - output_in_ratio
     -
        Specify whether to output ratios in a way that each row is 1.
        
        If this is not checked, the number of data samples is output as-is as a value of each cell.

   * - output
     -
        Specify the name of the CSV file to output the cross tabulation results to.
        
        If cross tabulation is executed from Output Result of the Evaluation tab, the cross tabulation results are saved to the specified file in the training result folder.


