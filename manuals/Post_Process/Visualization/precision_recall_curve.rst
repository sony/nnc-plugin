Precision/Recall curve
~~~~~~~~~~~~~~~~~~~~~~~~~~

This plugin draws a two-dimensional plot showing the tradeoff between precision and recall for different thresholds.
Higher AUC-PR score indicates a better model. By Decrease the threshold value, get more TP. 
If the predicted label is greater than the threshold, then it is classified as a positive prediction; otherwise it is a negative prediction.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     - Dataset CSV file. Output results are shown in the 'Evaluation' tab. 
       Default `output_result.csv` is used as input for drawing of `PR curve`. 

   * - target_variable
     - Target label in csv file.

   * - output_variable
     - Predicted label in csv file. For multiclassification give the variables as(y'__0,y'__1, ... , y'__n)

   * - width
     - Plot width, to be drawn in inches.

   * - height
     - Plot height, to be drawn in inches.

