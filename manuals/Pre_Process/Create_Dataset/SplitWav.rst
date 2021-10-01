Split Wav
~~~~~~~~~

Split the long Wav file contained in the dataset CSV file into small Wav files, and create a dataset CSV file with each small Wav file as data. When the original long Wav file cannot be processed by one neural network in the signal processing or segmentation task of the sound, processing can be executed by split the Wav file into small Wav files using this plug-in.

.. image:: figures/SplitWav1.png
   :align: center


.. list-table::
   :widths: 30 70
   :class: longtable

   * - input-csv
     - Specify the dataset CSV file to convert from

   * - input-variable1
     -
        Specify the variable name of the wav to be split from the variables included in the dataset CSV file.
        
        To split multiple variables containing Wav files of the same size into small Wav files, specify multiple variable names separated by commas.

   * - window-size1
     - Specify the size of the wav file after splittubg in samples

   * - overlap-size1
     - Specify the size to overlap with the next wav file in samples

   * -
        input-variable2
        
        window-size2
        
        overlap-size2
     - These settings are used when one dataset CSV file contains multiple wav files to be split and wav splitting is performed with settings different from the variables specified in input-variable1.

   * - output-dir
     - Specify the output folder for the converted dataset CSV file

   * - shuffle
     -
        Specify whether to shuffle each line of the converted dataset CSV file.
        
        true: shuffle randomly
        
        false: do not shuffle

   * -
        output_file1
        
        output_file2
     - Specify the name of the dataset CSV file to be created.

   * -
        ratio1
        
        ratio2
     -
        The sum of the ratio1 and ratio2 must be 100 (%).
        
        If ratio2 is 0, all data will be output to output_file1.


This plug-in outputs the following variables required for image recomposition to the output dataset CSV file.



.. list-table::
   :widths: 30 70
   :class: longtable

   * - index
     - Index of data in the source dataset CSV file

   * - original_length
     - Length of source Wav file (output for each variable)

   * - window_size
     - Window size (output for each variable)

   * - pos
     - Position of the split Wav file in the source Wav file (output for each variable)


.. image:: figures/SplitWav2.png
   :align: center


**Reference**

Wav file split using this plug-in can be restored to the original Wav file with the Restore Split Wav plug-in provided as a post-processing plug-in.

The lengths of the Wav files contained in the source dataset CSV file do not have to be the same.

When splitting the Wav file, this plug-in pads the range of the output Wav file that is not included in the original Wav file with 0.
