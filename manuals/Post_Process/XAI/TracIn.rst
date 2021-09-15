XAI/TracIn
~~~~~~~~~~

Using a method called TracIn, the influence of the input images on recognition result are evaluated. The dataset and the scores are shown in the influential order, which can be referred for data cleansing. This plugin runs when using GPU.

Estimating Training Data Influence by Tracing Gradient Descent
   - Garima Pruthi, Frederick Liu, Mukund Sundararajan, Satyen Kale
   - https://arxiv.org/abs/2002.08484

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


