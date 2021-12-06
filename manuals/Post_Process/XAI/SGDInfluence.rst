XAI/SGD Influence
~~~~~~~~~~~~~~~~~

Using a method called SGD Influence, the influence of the input images on recognition result is evaluated. The dataset and the scores are shown in the influential order, which can be referred for data cleansing.

The SGD Influence calculation in this plugin uses an approximate version of algorithm based on the following three papers.

* `Kenji Suzuki, Yoshiyuki Kobayashi, Takuya Narihira. "Data Cleansing for Deep Neural Networks with Storage-efficient Approximation of Influence Functions". , 2021. <https://arxiv.org/abs/2103.11807>`_
* `Satoshi Hara, Atsushi Nitanda, Takanori Maehara. "Data Cleansing for Models Trained with SGD". Advances in Neural Information Processing Systems 32, pages 4215–4224, 2019. <https://papers.nips.cc/paper/2019/hash/5f14615696649541a025d3d0f8e0447f-Abstract.html>`_
* `Pang Wei Koh, Percy Liang. "Understanding black-box predictions via influence functions". Proceedings of the 34th International Conference on Machine Learning, 2017 <http://proceedings.mlr.press/v70/koh17a>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input-train
     - Specify the dataset CSV file containing image files for which SGD Influence scores are calculated.

   * - input-val
     - Specify the dataset CSV file containing image files with which SGD Influence scores are calculated. This input-val dataset are used for SGD Influence scores calculation in accordance with input-train dataset, although the target of scoring is input-train dataset only. Specify the CSV file with different datasets other than input-train.

   * - output
     - Specify the name of the CSV file to output the inference results to.

   * - seed
     - Specify the random seed number to shuffle input-train data.

   * - model
     - Specify the model file (*.nnp) that will be used in the SGD Influence computation. To perform SGD Influence based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - batch_size
     - Specify the batch size to train with the model used in SGD Influence.

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


XAI/SGD Influence (tabular)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using a method called SGD Influence, the influence of the features in input table data on classification result is evaluated. The dataset index and the scores are shown in the influential order, which can be referred for data cleansing.

**NOTICE** *This plugin can be used for models without dropout layer since grad calculation is not available for the moment.*

`Satoshi Hara, Atsushi Nitanda, Takanori Maehara. "Data Cleansing for Models Trained with SGD". Advances in Neural Information Processing Systems 32, pages 4215–4224, 2019. <https://papers.nips.cc/paper/2019/hash/5f14615696649541a025d3d0f8e0447f-Abstract.html>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     - Specify the model file (*.nnp) that will be used in the SGD Influence computation. To perform SGD Influence based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - batch_size
     - Specify the batch size to train with the model used in SGD Influence.

   * - input-train
     - Specify the dataset CSV file containing the tabular data for which SGD Influence scores are calculated.

   * - input-val
     - Specify the dataset CSV file containing the tabular data with
       which SGD Influence scores are calculated. This input-val
       dataset are used for SGD Influence scores calculation in
       accordance with input-train dataset, although the target of
       scoring are input-train dataset only. Specify the CSV file with
       different datasets other than input-train.

   * - output
     - Specify the name of the CSV file to output the inference results to.

   * - seed
     - Specify the random seed number to shuffle input-train data.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.
The other columns than listed below are the same meaning as those in output_result.csv file that is generated as a result of evaluation.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - index
     - The index of the target instance in input-train dataset CSV file. Use this index to retrieve the order of rows as in input-train dataset CSV file.

   * - influence
     - The influence of the target instance. The order of rows in the output CSV file is sorted with this influence.
