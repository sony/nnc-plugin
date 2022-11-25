Precision/Recall curve
~~~~~~~~~~~~~~~~~~~~~~~~~~

This plugin draws a two-dimensional plot showing the tradeoff between precision and recall for different thresholds Higher AUC-PR score indicates a better model. Baseline of the PR curve is the horizontal line that indicates the value of the positive rate P/(P+N) — the smallest value of precision.


.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     - Dataset CSV file. Output results are shown in the 'Evaluation' tab. 
       Default `output_result.csv` is used as input for drawing of `PR curve`. 

   * - target_variable
     - Target label in csv file

   * - output_variable
     - Predicted label in csv file

   * - width
     - Plot width, to be drawn in inches

   * - height
     - Plot height, to be drawn in inches

   * - threshold
     - Threshold stepsize between 0 & 1.
       Get N number of threshold values between 0 & 1 by stepsize.

