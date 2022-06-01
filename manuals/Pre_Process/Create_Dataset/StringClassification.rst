String Classification
~~~~~~~~~~~~~~~~~~~~~

Converts to Neural Network Console dataset CSV file format based on a CSV file with the first column being a string and the second column being a category index. The Simple Text Classification plugin treats input sentences as word strings and indexes them word by word, while the String Classification plugin indexes characters by character, so it can be used for sentence classification that handles all languages.

The converted dataset CSV file will be the original strings converted to the character index sequence and its length.

The CSV file input by this plugin has almost the same format as the dataset CSV file of Neural Network Console. The first line is the header, and the second and subsequent lines are the data. In the dataset CSV file, enter the string as it is in each cell from the second row on the first column.

.. code::

   x:input_text,y:label
   (^^),0
   (;_;),1
   :-),0
   (T_T),1

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     - Specify the source CSV file.

   * - encoding
     -
        Specify the encoding of the input CSV file from the following.
        
        ascii: ASCII
        
        shift-jis: Shift-JIS
        
        utf-8: UTF-8
        
        utf-8-sig: UTF-8 (with BOM)

   * - max-length
     -
        Specify the maximum length of the input string.
        
        In the input string, the characters after the one specified by max-length are ignored.

   * - max-characters
     -
        Specify the maximum number of characters to index.
        
        Of the characters contained in the input CSV file, the characters that are 2 less than the number specified by max-characters in order from the frequently occurring characters will be indexed.
        
        Other characters are combined into one character called “Others”.

   * - min-occurrences
     -
        Specify the minimum frequency of occurrence of characters to index.
        
        Of the characters contained in the input CSV file, the characters that occur less than min-occurrences are combined into one character called “Others”.
        
        The final indexed number of characters will be the smaller of the numbers determined by max-characters and min-occurrences.

   * - normalize
     - Performs normalization processing to unify the same characters with different character codes.

   * - index-file-input
     -
        Specify the filename of an existing character index CSV file.
        
        A character index CSV file is a CSV file that consists of an index starting from 0 in the first column and characters in the second column, and each line consists of a character index and characters.
        
        When index-file-input is specified, instead of using the character index based on the input CSV file, the input string is converted to the character index series using the character index described in the character index CSV file specified here.

   * - index-file-output
     - Specify the file name of the character index CSV file to save the character index created based on the input CSV file.

   * - output-dir
     - Specify the folder to output the created dataset to.

   * - log-file-output
     - Specify the file name of the text file to save the log file displayed during conversion.

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
        Set the ratio of the data used in the dataset CSV file to be created.
        
        The sum of the ratio1 and ratio2 must be 100 (%).
        
        If ratio2 is 0, all data will be output to output_file1.


