Fairness/Equalised Odds (Tabular)
~~~~~~~~~~~~~~~~~~~~~~
Equalized Odds concept states that the model should correctly identify the positive outcome at equal rates across groups (same as in Equal Opportunity), but also miss-classify the positive outcome at equal rates across groups.
This means that we are only enforcing equality among individuals who reach similar outcomes.
This algorithm achieves one of the highest levels of algorithmic fairness.

This metric is computed as average of absolute difference between false positive rate and true positive rate for unprivileged and privileged groups.

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
     - Specify the name of the column in the input CSV file to use as the privileged variable (Class in the protected attribute with the majority is called privileged class)

   * - unprivileged_variable
     - Specify the name of the column in the input CSV file to use as the unprivileged variable (Class in the protected attribute with minority is called unprivileged class).

   * - clf_threshold
     - Specify the best optimal classification threshold.The default threshold for interpreting probabilities to class labels is 0.5.

   * - fair_threshold
     - Specify fairness threshold, between 0 & 1.0. Based on this value, model outputs whether outcome is "fair" or "unfair". Default value is 0.10. So, all outcomes between 0.0 and 0.1 are "fair".

   * - num_samples
     - Specify the number of samples to compute the `Equalised Odd`, by default num_samples is "all", which means compute the Equal opportunity  for all the samples in the input file.

   * - output
     - Specify the name of the CSV file to output the Equalised odd (AAOD) result.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Plot fairness check enables us to look how the fairness between privileged and unprivileged groups with respect to the fairness definition. If the bar plot reaches out of the green zone it means that for this subgroup fairness goal is not satisfied.

   * - Equalised Odds
     - Low Equalised Odds values mean a fair model - desirable, while high values mean the model is not fair.




