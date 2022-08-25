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

   * - label_variable
     - Specify the name of the column in the input CSV file to use as the label variable

   * - output_variable
     - Specify the name of the column in the input CSV file to use as the output variable(classification output, in this case, Sigmoid as output)

   * - privileged_variable
     - Specify the name of the column in the input CSV file to use as the privileged variable, to compute the discrimination score. Class in the protected attribute with the majority is called privileged class. By default, 'female' is the privileged variable.

   * - unprivileged_variable
     - Specify the name of the column in the input CSV file to use as the unprivileged variable, to compute the discrimination score. Class in the protected attribute with minority is called unprivileged class. By default, 'male' is the unprivileged variable.

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

   * - Object Variable
     - Specified label variable while computing the CV score.

   * - Output variable
     - Specified classifier output variable while computing the CV score.

   * - Privileged variable
     - Specified privileged variable to compute the discrimination score.

   * - Unprivileged variable
     - Specified unprivileged variable to compute the discrimination score.

   * - Number of samples
     - Number of samples taken to compute the CV score of the model.

   * - CV score
     - Low CV score values mean a fair model - desirable, while high values mean the model is not fair.

   * - Accuracy
     - High accuracy values mean a good performance model - desirable.



