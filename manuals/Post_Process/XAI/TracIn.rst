XAI/TracIn
~~~~~~~~~~

Using a method called TracIn, the influence of the input images on recognition result are evaluated. The dataset and the scores are shown in the influential order, which can be referred for data cleansing. This plugin runs when using GPU.  


`Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan. "Estimating Training Data Influence by Tracing Gradient Descent". In Advances in Neural Information Processing Systems, 2020. <https://papers.nips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf>`_


Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input-train
     - Specify the dataset CSV file containing image files for which TracIn scores are calculated.

   * - model
     - Specify the model used for TracIn calculation (resnet23 or resnet56).

   * - output
     - Specify the name of the CSV file to output the inference results to.

   * - train_batch_size
     - Specify the batch size to train with the model used in TracIn.

   * - train_epochs
     - Specify the epoch size to train with the model used in TracIn.

   * - seed
     - Specify the random seed number for data augmentation.


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

Link
========
| Please see below for the detailed explanation of algorithm.
| https://github.com/sony/nnabla-examples/tree/master/responsible_ai/tracin#overview


