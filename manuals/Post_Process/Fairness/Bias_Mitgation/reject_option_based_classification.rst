Reject Option-Based Classification (Tabular)
~~~~~~~~~~~~~~~~~~~~~~
Reject Option-Based Classification (ROC) is a post-processing techinque to mitigate bias at the model prediction stage, enhancing the favourable outcomes to unprivileged groups and unfavorable outcomes to privileged groups, in a confidence band around the decision boundary with highest uncertainty.

This plugin will estimate the `optimal classification threshold` and `margin for reject option classification` that optimizes the metric provided. Please refer to our colab tutorial notebook for more info about ROC algorithm : `ROC <https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/rejection_option_based_classification.ipynb#scrollTo=k_aleVIr6GeX>`_


Citation 
===================

`Kamiran, Faisal, Asim Karim, and Xiangliang Zhang. "Decision theory for discrimination-aware classification." In 2012 IEEE 12th International Conference on Data Mining, pp. 924-929. IEEE, 2012. <https://ieeexplore.ieee.org/document/6413831>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Dataset CSV file. output_result.csv is input for "ROC" computation

   * - target_variable
     - Target variable column name in input CSV file

   * - output_variable
     - Output variable column name in input CSV file (classification output: y' is default output variable)

   * - privileged_variable
     - Privileged variable name in input CSV file (Class in the protected attribute with majority is called the privileged class)

   * - unprivileged_variable
     - Unpriviliged variable name in input CSV file (Class in the protected attribute with minority is called the unprivileged class)

   * - fair_metric
     - Name of the fairness metric to be used for the optimization. Allowed options are “Demographic Parity”, “Equalised Odds”, “Equal Opportunity”. By default “Demographic Parity” fairness metric is used for the optimization.

   * - metric_ub
     - Upper bound of constraint on the fairness metric value. Between -1.0 and 1.0; default value is 0.10.
   
   * - metric_lb
     - Lower bound of constraint on the fairness metric value. Between -1.0 and 1.0; default value is -0.10.

   * - output
     - CSV file name to output the "ROC" result. "roc.csv" is default output file.

Output Information
===================

Result of this plugin is saved in the designated output path as CSV file.
Information on the columns of CSV file is as follows:

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Classification Threshold
     - Estimated optimal classification threshold for specified optimization metric. Use the estimated `classification threshold` for model prediction. 

   * - ROC Margin
     - Margin for ROC (critical region boundary) for specified optimization metric. Use `ROC Margin` for model prediction.
   
   * - Accuracy
     -  Accuracy of the model after application of ROC algorithm.
   
   * - Fairness metric
     - Model fairness for specified optimization metric.

Reject Option-Based Classification Predict (Tabular)
~~~~~~~~~~~~~~~~~~~~~~
This plugin	will obtain fair model predictions, based on ROC plugin results.

Note: Please run the "Reject Option-Based Classification" plugin before running this plugin.


Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Dataset CSV file with classification scores used to compute predicted labels. Default is output_result.csv.

   * - output_variable
     - Output variable column name in the input CSV file (classification output, in this case, Sigmoid).

   * - privileged_variable
     - Privileged variable column name in the input CSV file (class in the protected attribute with majority is called the privileged class).

   * - unprivileged_variable
     - Unprivileged variable column name in the input CSV file (class in the protected attribute with minority is called the unprivileged class).

   * - roc_params
     - "Reject Option-Based Classification" plugin processed output CSV file where the estmiated ROC params (classification threshold and ROC Margin) are saved. `roc.csv` is default file name. 

   * - output
     - Output fair predictions obtained by the ROC method to this file. Default is "roc_predict.csv".

Output Information
===================

The result of this plugin is saved in the designated output path as CSV file. 
The information on the columns of CSV file is as follows.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - ROC Predicted
     - New classification results are predicted using ROC method.



