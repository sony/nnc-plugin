Fairness/Theil index(Tabular)
~~~~~~~~~~~~~~~~~~~~~~
Theil index, Computed as the generalized entropy of benefit for all individuals in the dataset.
with alpha = 1. It measures the inequality in benefit allocation for individuals.

`Speicher, Till, Hoda Heidari, Nina Grgic-Hlaca, Krishna P. Gummadi, Adish Singla, Adrian Weller, and Muhammad Bilal Zafar. "A unified approach to quantifying algorithmic unfairness: Measuring individual &group unfairness via inequality indices." In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2239-2248. 2018. <https://dl.acm.org/doi/abs/10.1145/3219819.3220046>`_

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

   * - clf_threshold
     - Specify the best optimal classification threshold.The default threshold for interpreting probabilities to class labels is 0.5.

   * - fair_threshold
     - Specify fairness threshold, between 0 & 1.0. Based on this value, model outputs whether outcome is "fair" or "unfair". Default value is 0.10. So, all outcomes between 0.0 and 0.1 are "fair".

   * - num_samples
     - Specify the number of samples to compute the `Theil index`, by default num_samples is "all", which means compute the Theil index  for all the samples in the input file.

   * - output
     - Specify the name of the CSV file to output the `Theil index (TI)` result.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Plot fairness check enables us to look how the fairness in individuals in the dataset with respect to the fairness definition. If the bar plot reaches out of the green zone it means that for individual fairness goal is not satisfied.

   * - Theil index
     - Ideal value of this Theil index is 0. Low Theil index values mean a fair model - desirable, while high values mean the model is not fair.

