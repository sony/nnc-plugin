Fairness/CV Score (Tabular)
~~~~~~~~~~~~~~~~~~~~~~
CV Score measures the discrimination score of the model for tabula data, by subtracting the conditional probability of the positive outcome between protected and non protected members.


`Fairness-aware classifier with prejudice remover regularizer. Toshihiro Kamishima, Shotaro Akaho, Hideki Asoh & Jun Sakuma.Joint European Conference on Machine Learning and Knowledge Discovery in Databases ECML PKDD 2012: Machine Learning and Knowledge Discovery in Databases pp 35–50 <https://link.springer.com/chapter/10.1007/978-3-642-33486-3_3>`_


Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Specify the dataset CSV file containing the data to analyze. To compute the CV score, output result shown in the 'Evaluation' tab, use the default output_result.csv.

   * - target_variable
     - Specify the name of the column in the input CSV file to use as the target variable.

   * - output_variable
     - Specify the name of the column in the input CSV file to use as the output variable(classification output , by default y' as output variable)

   * - privileged_variable
     - Specify the name of the column in the input CSV file to use as the privileged variable, to compute the discrimination score. Class in the protected attribute with the majority is called privileged class.

   * - unprivileged_variable
     - Specify the name of the column in the input CSV file to use as the unprivileged variable, to compute the discrimination score. Class in the protected attribute with minority is called unprivileged class.

   * - clf_threshold
     - Specify the best optimal classification threshold.The default threshold for interpreting probabilities to class labels is 0.5.

   * - fair_threshold
     - Specify fairness threshold, between 0 & 1.0. Based on this value, model outputs whether outcome is "fair" or "unfair". Default value is 0.10. So, all outcomes between 0.0 and 0.1 are "fair".

   * - num_samples
     - Specify the number of samples to compute the CV score, by default num_samples is "all", which means compute the CV score of all the samples in the input file.

   * - output
     - Specify the name of the CSV file to output the CV score & accuracy result to.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Plot fairness check enables us to look how the fairness between privileged and unprivileged groups with respect to the fairness definition. If the bar plot reaches out of the green zone it means that for this subgroup fairness goal is not satisfied.

   * - CV score
     - Low CV score values mean a fair model - desirable, while high values mean the model is not fair.

   * - Accuracy
     - High accuracy values mean a good performance model - desirable.



