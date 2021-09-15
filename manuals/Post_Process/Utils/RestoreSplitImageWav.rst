Utils/Restore Split Image
~~~~~~~~~~~~~~~~~~~~~~~~~

Restores the original high resolution image from images split into multiple small patches for processing with a neural network. It can be used to restore high-resolution images from split images with the Split Image plug-in or the processing results.

.. image:: figures/RestoreSplitImage.png
   :align: center

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input-csv
     -
        Specify a dataset CSV file containing patch images used to restore images.
        
        The dataset CSV file specified here must contain various variables specified by the following arguments.

   * - image_index
     - Specify a variable that indicates the index of the image to be restored from the variables included in the dataset CSV file.

   * - input-variable
     - Specify a variable that indicates the patch image used to restore the image from the variables included in the dataset CSV file.

   * - height-variable
     - Specify a variable that indicates the vertical size of the image to be restored from the variables included in the dataset CSV file.

   * - width-variable
     - Specify a variable that indicates the horizontal size of the image to be restored from the variables included in the dataset CSV file.

   * - patch-size-variable
     - Specify a variable that indicates the vertical and horizontal size of the patch image from the variables included in the dataset CSV file.

   * - overlap-size-variable
     - Specify a variable that indicates the size of the patch image that overlaps with next patches from the variables included in the dataset CSV file.

   * - top-variable, left-variable
     - Specify variables that indicate the Y and X coordinates of the upper left coordinate of the image to be restored from the variables included in the dataset CSV file.

   * - output-csv
     - Specify the file name of the CSV file to output the restoration result.

   * - inherit-cols
     - Specify columns to be inherited from the input CSV file specified by input-csv to the output CSV file specified by output-csv with the variable name.


**Reference**

For each variable such as patch-size, overlap-size, top and left, please also refer to the explanation of the Split Image plug-in.

Utils/Restore Split Wav
~~~~~~~~~~~~~~~~~~~~~~~

Restores the original long Wav file from Wav files split into multiple small waveforms for processing with a neural network. It can be used to restore long Wav files from split Wav files with the Split Wav plug-in or the processing results.

.. image:: figures/RestoreSplitWav.png
   :align: center

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input-csv
     -
        Specify a dataset CSV file containing small Wav files used to restore Wav files.
        
        The dataset CSV file specified here must contain various variables specified by the following arguments.

   * - wav_index-variable
     - Specify a variable that indicates the index of the Wav file to be restored from the variables included in the dataset CSV file.

   * - input-variable
     - Specify a variable that indicates the small Wav files used to restore the long Wav files from the variables included in the dataset CSV file.

   * - length-variable
     - Specify a variable that indicates the length of the Wav file to be restored from the variables included in the dataset CSV file.

   * - window-size-variable
     - Specify a variable that indicates the length of the small Wav file from the variables included in the dataset CSV file.

   * - overlap-size-variable
     - Specify a variable that indicates the length of overlaps with the next Wav file from the variables included in the dataset CSV file.

   * - pos-variable
     - Specify variables that indicate the position of the waveform to be restored from the variables included in the dataset CSV file.

   * - crossfade
     -
        By checking crossfade, linear crossfade processing will be performed for the part where the splited waveform overlaps with the next waveform.
        
        Uncheck crossfade to combine using only the data in the window-size – overlap-size \* 2 range of the splited Wav files.

   * - default-sampling-freq
     -
        Specify the sampling frequency of the output Wav file used when the variable specified by input-variable is a data CSV file.
        
        If the variable specified by input-variable is a Wav file, the sampling frequency of the output Wav file will be the same as the sampling frequency of the splited Wav file.

   * - output-csv
     - Specify the file name of the CSV file to output the restoration result.

   * - inherit-cols
     - Specify columns to be inherited from the input CSV file specified by input-csv to the output CSV file specified by output-csv with the variable name.


**Reference**

For each variable such as window-size, overlap-size and pos, please also refer to the explanation of the Split Wav plug-in.

