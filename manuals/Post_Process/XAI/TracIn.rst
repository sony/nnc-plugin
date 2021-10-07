XAI/TracIn
~~~~~~~~~~

Using a method called TracIn, the influence of the input images on recognition result are evaluated. The dataset and the scores are shown in the influential order, which can be referred for data cleansing. This plugin runs when using GPU.
Please see below for the detailed explanation of algorithm.
https://github.com/sony/nnabla-examples/tree/master/responsible_ai/tracin#overview


Estimating Training Data Influence by Tracing Gradient Descent
   - Garima Pruthi, Frederick Liu, Mukund Sundararajan, Satyen Kale
   - https://papers.nips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf

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


