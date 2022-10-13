Fairness/Demographic(statistical) Parity Difference(Tabular)
~~~~~~~~~~~~~~~~~~~~~~
Demographic Parity, also called Independence or Statistical Parity, is a well-known criteria to calculate fairness in machine learning.
According to it, proportions of all segments of protected class (eg., gender) should receive equal rates of positive outcome.
For example, the probability of getting admission to a college must be independent of gender. In simple terms, outcome must be independent of protected class.

This metric is computed as the difference between the rate of positive outcomes in unprivileged and privileged groups.

`Sam Corbett-Davies, Emma Pierson, Avi Feller, Sharad Goel, and Aziz Huq. 2017.Algorithmic decision making and the cost of fairness. In Proceedings of KDD <https://dl.acm.org/doi/abs/10.1145/3097983.3098095>`_

`Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. 2012. Fairness through awareness. In Proceedings of the 3rd Innovations in Theoretical Computer Science Conference (ITCS). 214–226<https://dl.acm.org/doi/abs/10.1145/2090236.2090255>`_

`Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan. 2017. Inherent Trade-Offs in the Fair Determination of Risk Scores. In Proceedings of ITCS.<https://arxiv.org/abs/1609.05807>`_

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

   * - fair_threshold
     - Specify fairness threshold, between -1.0 & 1.0. Based on this value, model outputs whether outcome is "fair" or "unfair". Default value is 0.10. So, all outcomes between -0.1 and 0.1 are "fair".

   * - num_samples
     - Specify the number of samples to compute the demographic parity, by default num_samples is "all", which means compute the demographic parity for all the samples in the input file.

   * - output
     - Specify the name of the CSV file to output the Demographic parity(DPD) result.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - Fairness Plot
     - Plot fairness check enables us to look how the fairness between privileged and unprivileged groups with respect to the fairness definition. If the bar plot reaches out of the green zone it means that for this subgroup fairness goal is not satisfied.

   * - Demographic Parity
     - The ideal value of this Demographic Parity metric is 0, value of Demographic Parity < 0 implies a higher benefit for the privileged group, value of Demographic Parity > 0 implies a higher benefit for the unprivileged group.

