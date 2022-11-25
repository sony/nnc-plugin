Fairness/Disparate Impact (Tabular)
~~~~~~~~~~~~~~~~~~~~~~
Disparate Impact is a metric to evaluate fairness. It compares the proportion of individuals that receive positive output for two groups: an unprivileged group and a privileged group.

This metric computed as the ratio of rate of favorable outcome for the unprivileged group to that of the privileged group.

`Michael Feldman, Sorelle A. Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubramanian. "Certifying and removing disparate impact." In proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining, pp. 259-268. 2015. <https://arxiv.org/abs/1412.3756v3>`_

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

   * - output_variable
     - Specify the name of the column in the input CSV file to use as the output variable(classification output , by default y' as output variable)

   * - privileged_variable
     - Specify the name of the column in the input CSV file to use as the privileged variable (class in the protected attribute with the majority is called privileged class)

   * - unprivileged_variable
     - Specify the name of the column in the input CSV file to use as the unprivileged variable (class in the protected attribute with minority is called unprivileged class).

   * - clf_threshold
     - Specify the best optimal classification threshold.The default threshold for interpreting probabilities to class labels is 0.5.

   * - num_samples
     - Specify the number of samples to compute the `Disparate Impact`, by default num_samples is "all", which means compute the `Disparate Impact` for all the samples in the input file.

   * - output
     - Specify the name of the CSV file to output the Disparate Impact(DI) result.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Plot fairness check enables us to look how the fairness between privileged and unprivileged groups with respect to the fairness definition. If the bar plot reaches out of the green zone it means that for this subgroup fairness goal is not satisfied.

   * - Disparate Impact
     - The ideal value of this Disparate Impact metric is 1.0 , value of Disparate Impact < 1.0  implies a higher benefit for the privileged group, and value of Disparate Impact > 1.0  implies a higher benefit for the unprivileged group.



