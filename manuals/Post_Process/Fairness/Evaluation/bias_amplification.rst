Fairness/Bias Amplification
~~~~~~~~~~~~~~~~~~~~~~
Bias amplification measures how much more often a target attribute is predicted with a protected attribute than the ground truth value.

`Wang, Angelina, and Olga Russakovsky. "Directional bias amplification." In International Conference on Machine Learning, pp. 10882-10893. PMLR, 2021. <https://arxiv.org/pdf/2102.12594.pdf>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input
     - Specify the dataset CSV file containing the data to analyze. Output result are shown in the 'Evaluation' tab. Default output_result.csv file acts as input for Bias Amplification computation.

   * - target_variable
     - Specify the name of the column in the input CSV file to use as the target variable.

   * - output_variable
     - Specify the name of the column in the input CSV file to use as the output variable (classification output, by default y' is the output variable).

   * - privileged_variable
     - Specify the name of the column in the input CSV file to use as the privileged variable (Class in the protected attribute with the majority is called privileged class).

   * - unprivileged_variable
     - Specify the name of the column in the input CSV file to use as the unprivileged variable (Class in the protected attribute with minority is called unprivileged class).

   * - clf_threshold
     - Specify the optimal classification threshold. Default threshold for interpreting probabilities to class labels is 0.5.

   * - fair_threshold
     - Specify fairness threshold, between -1.0 & 1.0. Based on this value, model outputs whether outcome is "fair" or "unfair". Default value is 0.10. So, all outcomes between -0.1 and 0.1 are "fair".

   * - num_samples
     - Specify the number of samples to compute the `Bias Amplification`. By default, num_samples is "all", which leads to computation of Bias Amplification for all samples in the input file.

   * - output
     - Specify the name of the CSV file to output the Bias Amplification (BA) result.

Output Information
===================

Result of this plugin is saved in the designated 'output' path as CSV file.
Information on the columns of CSV file is as follows:

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Fairness plot helps in visualization of adherence/deviation of privileged and unprivileged groups with respect to the fairness definition. If the bar plot stretches beyond green zone, that is indication of non-satisfaction of fairness goal for the corresponding sub-group. 

   * - Bias Amplification
     - Low Bias Amplification values mean fair model - desirable; high values imply unfair model.
