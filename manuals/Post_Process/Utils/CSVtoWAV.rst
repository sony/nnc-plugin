Utils/CSV to Wav (batch)
~~~~~~~~~~~~~~~~~~~~~~~~

Converts the data CSV files contained in the dataset CSV file into Wav files.



.. list-table::
   :widths: 30 70
   :class: longtable

   * - input-csv
     - Specify a dataset CSV file that contains the CSV files to be converted into Wav files in the cell.

   * - input-variable
     -
        Specify the variables of the CSV file to be converted to the Wav file from the variables included in the dataset CSV file.
        
        The matrix size of each data CSV file contained in the specified variable must be "length, number of channels"

   * - Sampling_rate
     - Specify the sampling frequency of the converted Wav file

   * - output-csv
     - Specify the file name of the converted dataset CSV file


**Reference**

The converted Wav files are saved in the “wavfiles” folder created in the folder specified by output-csv.

The bit length of the converted Wav file is 16 bits. The amplitude of the converted Wav file is the value written in the CSV file multiplied by 32,768 and converted to an integer.

