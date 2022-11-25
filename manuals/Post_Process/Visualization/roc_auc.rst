ROC and AUC curve
~~~~~~~~~~~~~~~~~~~~~~~~~~

This plugin draws a two-dimensional plot between TPR and FPR at different thresholds.
Higher AUC score indicates better model. 

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     - Dataset CSV file. Output results are shown in the 'Evaluation' tab. 
       Default `output_result.csv` is used as input for drawing of `ROC and AUC curve`. 

   * - target_variable
     - Target label in csv file.

   * - output_variable
     - Predicted label in csv file.

   * - width
     - Plot width, to be drawn in inches.

   * - height
     - Plot height, to be drawn in inches.

   * - threshold
     - Threshold stepsize between 0 & 1 (default step = 0.02).
       Get N number of threshold values between 0 & 1.




 


