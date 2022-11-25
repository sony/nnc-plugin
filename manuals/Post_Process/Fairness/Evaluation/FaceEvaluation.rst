XAI/Face evaluation
~~~~~~~~~~~~~~~~~~~

Measure skin color of human face in input images, calculating a score called Individual Typology Angle (ITA).

`Michele Merler, Nalini Ratha, Rogerio S Feris, John R Smith. "Diversity in faces". Computer Vision and Pattern Recognition, 2019. <https://arxiv.org/abs/1901.10436>`_

Input Information
===================

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     - Specify the dataset CSV file containing image files to be processed by face evaluation.

   * - output
     - Specify the name of the CSV file to output the inference results to.

Output Information
===================

The result of this plugin is saved in the designated 'output' path as CSV file.
The information on the columns of CSV file is as follows.
The other columns than listed below are the same meaning as those in output_result.csv file that is generated as a result of evaluation.

.. list-table::
   :widths: 30 70
   :class: longtable

   * - ITA
     - The ITA score for the target instance. High ITA values mean fair skin, while low values means dark skin.
