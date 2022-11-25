Fairness/Equal Opportunity Difference(Tabular)
~~~~~~~~~~~~~~~~~~~~~~
Equal opportunity difference (EOD) concept states that the model should correctly identify the positive outcome at equal rates across groups (matching the true positive rates for different values of the protected attribute), assuming that people in each group qualify for it.

This metric is computed as the difference between true positive rate (true positives / positives) between the unprivileged and the privileged groups.

`Moritz Hardt, Eric Price, and Nati Srebro. "Equality of opportunity in supervised learning." Advances in neural information processing systems 29 (2016) <https://arxiv.org/pdf/1610.02413.pdf>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Specify the dataset CSV file containing the data to analyze. Output result shown in the 'Evaluation' tab, use the default output_result.csv.

   * - target_variable
     - Specify the name of the column in the input CSV file to use as the target variable.

   * - output_variable
     - Specify the name of the column in the input CSV file to use as the output variable(classification output , by default y' as output variable)

   * - privileged_variable
     - Specify the name of the column in the input CSV file to use as the privileged variable (class in the protected attribute with the majority is called privileged class)

   * - unprivileged_variable
     - Specify the name of the column in the input CSV file to use as the unprivileged variable (class in the protected attribute with minority is called unprivileged class).

   * - clf_threshold
     - Specify the best optimal classification threshold.The default threshold for interpreting probabilities to class labels is 0.5.

   * - fair_threshold
     - Specify fairness threshold, between -1.0 & 1.0. Based on this value, model outputs whether outcome is "fair" or "unfair". Default value is 0.10. So, all outcomes between -0.1 and 0.1 are "fair".

   * - num_samples
     - Specify the number of samples to compute the `Equal opportunity`, by default num_samples is "all", which means compute the Equal opportunity  for all the samples in the input file.

   * - output
     - Specify the name of the CSV file to output the Equal opportunity difference (EOD) result.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Plot fairness check enables us to look how the fairness between privileged and unprivileged groups with respect to the fairness definition. If the bar plot reaches out of the green zone it means that for this subgroup fairness goal is not satisfied.

   * - Equal Opportunity
     - Ideal value of this Equal Opportunity metric is 0. Value of Equal Opportunity < 0 implies higher benefit for the privileged group, while value of Equal Opportunity > 0 implies higher benefit for the unprivileged group.




