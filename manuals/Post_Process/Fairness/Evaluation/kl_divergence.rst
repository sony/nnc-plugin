Fairness/KL Divergence
~~~~~~~~~~~~~~~~~~~~~~
KL Divergenve metric, also called `Threshold Invariant` fairness metric, enforces equitable performances across different groups independent of the decision threshold. 

`Chen, Mingliang, and Min Wu. "Towards threshold invariant fair classification." In Conference on Uncertainty in Artificial Intelligence, pp. 560-569. PMLR, 2020. <https://arxiv.org/pdf/2006.10667.pdf>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Specify the dataset CSV file containing the data to analyze. Output result shown in the 'Evaluation' tab. Default output_result.csv is used as input for computation of KL Divergence measure.

   * - target_variable
     - Specify the name of the column in the input CSV file to use as the target variable.

   * - output_variable
     - Specify the name of the column in the input CSV file to use as the output variable (classification output, by default y' is output variable)

   * - privileged_variable
     - Specify the name of the column in the input CSV file to use as the privileged variable (Class in the protected attribute with the majority is called privileged class).

   * - unprivileged_variable
     - Specify the name of the column in the input CSV file to use as the unprivileged variable (Class in the protected attribute with minority is called unprivileged class).

   * - fair_threshold
     - Specify fairness threshold, between -1.0 & 1.0. Based on this value, model outputs whether outcome is "fair" or "unfair". Default value is 0.10. So, all outcomes between -0.1 and 0.1 are "fair".

   * - num_samples
     - Specify the number of samples to compute the `KL Divergence`. By default num_samples is "all", which leads to computation of KL Divergence for all the samples in the input file.

   * - output
     - Specify the name of the CSV file to output the KL Divergence (KL) result.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Fairness plot helps in visualization of adherence/deviation of privileged and unprivileged groups with respect to the fairness definition. If the bar plot stretches beyond green zone, that is indication of non-satisfaction of fairness goal for the corresponding sub-group. 

   * - KL Divergence
     - Low KL Divergence values indicate a fair model - desirable; high values indicate possibility of lack of fairness.
