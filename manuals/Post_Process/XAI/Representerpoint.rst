XAI/RepresenterPointSelection
~~~~~~~~~~

Using a method called Representer Point Selection, the influence of the training images on inference the test images are evaluated. 
The top-k influential samples are shown for a test data, which can be referred for understanding the basis for prediction. 
This plugin runs when using GPU.
Please see below for the detailed explanation of algorithm.
https://github.com/sony/nnabla-examples/tree/master/responsible_ai/representer_point#representer-point-selection

Representer Point Selection for Explaining Deep Neural Networks
   - Chih-Kuan Yeh, Joon Sik Kim, Ian E.H. Yen, Pradeep Ravikumar
   - https://proceedings.neurips.cc/paper/2018/file/8a7129b8f3edd95b7d969dfc2c8e9d9d-Paper.pdf

.. list-table::
   :widths: 30 70
   :class: longtable

   * - top_k
     - Specify the number of top-k influential training samples which are presented for a test sample.

   * - num-samples
     - Specify the number of presented test images.

   * - output
     - Specify the name of the CSV file to output the inference results to.
    
   * - model
     - Specify the pretrained model path (nnp file).

   * - input-train
     - Specify the training dataset CSV file containing image files for which influences are calculated.

   * - input-val
     - Specify the validation dataset CSV file containing image files for which influences are calculated.

   * - normalize
     - Specify the image normaliztion. 
     
   * - lmbd
     - Specify weight factor of l2.
   
   * - epoch
     - Specify the epochs of finetuning.


Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - test_sample
     - The test sample images

   * - label; pred
     - The test sample's label and predicted label. (label index)

   * - positive_x
     - the best (top-x) influential samples for prediction of test data.

   * - positive_x
     - the worst (top-x) influential samples for prediction of test data.

Link
========
| Please see below for the detailed explanation of algorithm.
| https://github.com/sony/nnabla-examples/tree/master/responsible_ai/representer_point#representer-point-selection


