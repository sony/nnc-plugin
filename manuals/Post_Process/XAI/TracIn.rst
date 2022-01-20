XAI/TracIn
~~~~~~~~~~

<<<<<<< HEAD
<<<<<<< HEAD
Using a method called TracIn, the influence of the input images on recognition result are evaluated. The dataset and the scores are shown in the influential order, which can be referred for data cleansing. This plugin runs when using GPU.
Please see below for the detailed explanation of algorithm.
https://github.com/sony/nnabla-examples/tree/master/responsible_ai/tracin#overview


Estimating Training Data Influence by Tracing Gradient Descent
   - Garima Pruthi, Frederick Liu, Mukund Sundararajan, Satyen Kale
   - https://papers.nips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf
=======
<<<<<<< HEAD
Using a method called TracIn, the influence of the input images on recognition result are evaluated. The dataset and the scores are shown in the influential order, which can be referred for data cleansing. This plugin runs when using GPU.  


`Garima Pruthi, Frederick Liu, Satyen Kale, Mukund Sundararajan. "Estimating Training Data Influence by Tracing Gradient Descent". In Advances in Neural Information Processing Systems, 2020. <https://papers.nips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf>`_


Input Information
===================
=======
=======
>>>>>>> d2c937e (Update TracIn.rst)
Using a method called TracIn, the influence of the input images on recognition result are evaluated. The dataset and the scores are shown in the influential order, which can be referred for data cleansing. This plugin runs when using GPU.
Please see below for the detailed explanation of algorithm.
https://github.com/sony/nnabla-examples/tree/master/responsible_ai/tracin#overview


Estimating Training Data Influence by Tracing Gradient Descent
   - Garima Pruthi, Frederick Liu, Mukund Sundararajan, Satyen Kale
   - https://papers.nips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf
<<<<<<< HEAD
>>>>>>> 0242433 (added links and change paper link)
>>>>>>> 53d54ed (added links and change paper link)
=======
>>>>>>> d2c937e (Update TracIn.rst)

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input-train
     - Specify the dataset CSV file containing image files for which TracIn scores are calculated.

   * - model-path
     - Specify the pretrained model path (nnp file).

   * - output
     - Specify the name of the CSV file to output the inference results to.
     
   * - normalize
     - Specify the image normaliztion. (True of False)
   
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
   

   * - x:image
     - The image path of the training data.
     
   * - y:label
     - The randomly shuffled label of the training data.
     
   * - original_label
     - The true (non-shuffled) label of the training data.
   

Link
========
| Please see below for the detailed explanation of algorithm.
| https://github.com/sony/nnabla-examples/tree/master/responsible_ai/tracin#overview


