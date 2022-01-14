Influence Functions
~~~~~~~~~~~~~~~~~~~

Using a method called Influence Functions, the influence of the input
images on recognition result are evaluated. The dataset and the scores
are shown in the influential order, which can be referred for data
cleansing.

`Pang Wei Koh, Percy Liang. "Understanding black-box predictions via influence functions". Proceedings of the 34th International Conference on Machine Learning, 2017 <http://proceedings.mlr.press/v70/koh17a>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable
   :header-rows: 1

   * - Property
     - Notes

   * - input-train
     - Specify the dataset CSV file containing image files for which
       Influence Functions scores are calculated.

   * - input-val
     - Specify the dataset CSV file containing image files with which
       Influence Functions scores are calculated. This input-val
       dataset are used for Influence Functions scores calculation in
       accordance with input-train dataset, although the target of
       scoring are input-train dataset only. Specify the CSV file with
       different datasets other than input-train.

   * - output
     - Specify the name of the CSV file to output the inference results to.

   * - n_trials
     - Specify the number of trials to shuffle input-train data and to calculate the mean value of influence results.

   * - model
     - Specify the model file (*.nnp) that will be used in the
       Influence Functions computation. To perform Influence Functions
       based on the training result selected in the Evaluation tab,
       use the default results.nnp.

   * - batch_size
     - Specify the batch size to train with the model used in
       Influence Functions.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.
The other columns than listed below are the same meaning as those in output_result.csv file that is generated as a result of evaluation.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - influence
     - The influence of the target instance. The order of rows in the output CSV file is sorted with this influence.

   * - datasource_index
     - The index of the target instance in input-train dataset CSV file. Use this index to retrieve the order of rows as in input-train dataset CSV file.
