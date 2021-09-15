Simple Text Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts to Neural Network Console dataset CSV file format based on a CSV file with the first column being an English string and the second column being a category index. The converted dataset CSV file will be the original English strings converted to the word index sequence and its length.

The CSV file input by this plugin has almost the same format as the dataset CSV file of Neural Network Console. The first line is the header, and the second and subsequent lines are the data. In the dataset CSV file, enter the English text as it is in each cell from the second row on the first column.

.. code:: csv

        x:input_text,y:label
        Tomorrow's weather is sunny.,0
        This is a pen.,1
        The weather the day after tomorrow is rainy.,0
        This is an apple.,1

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
        Specify the maximum word length of the input string.
        
        In the input string, the words after the one specified by max-length are ignored.

   * - max-words
     -
        Specify the maximum number of words to index.
        
        Of the words contained in the input CSV file, the words that are 2 less than the number specified by max-words in order from the frequently occurring words will be indexed.
        
        Other words are combined into one word called “Others”.

   * - min-occurences
     -
        Specify the minimum frequency of occurrence of words to index.
        
        Of the words contained in the input CSV file, the words that occur less than min-occurences are combined into one word called “Others”.
        
        The final indexed number of words will be the smaller of the numbers determined by max-words and min-occurences.

   * - normalize
     - Performs normalization processing to unify the same characters with different character codes.

   * - index-file-input
     -
        Specify the filename of an existing word index CSV file.
        
        A word index CSV file is a CSV file that consists of an index starting from 0 in the first column and words in the second column, and each line consists of a word index and words.
        
        When index-file-input is specified, instead of using the word index based on the input CSV file, the input string is converted to the word index series using the word index described in the word index CSV file specified here.

   * - index-file-output
     - Specify the file name of the word index CSV file to save the word index created based on the input CSV file.

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


**Reference**

The text classification dataset created using this plugin can be trained by using the 20newsgroups_classification sample project.

For training, set the Size property of Input layer to the value specified by max-length, set NumClass property of Embed layer to number of words determined by max-words or m in-occurences, set OutShape property of Affine layer to number of classification classes, respectively.


Simple Japanese Text Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the Japanese version of Simple Text Classification.

Converts to Neural Network Console dataset CSV file format based on a CSV file with the first column being a Japanese string and the second column being a category index. The converted dataset CSV file will be the original Japanese strings converted to the word index sequence and its length.

The CSV file input by this plugin has almost the same format as the dataset CSV file of Neural Network Console. The first line is the header, and the second and subsequent lines are the data. In the dataset CSV file, enter the Japanese text as it is in each cell from the second row on the first column.

.. code:: csv

        x:input_text,y:label
        明日の天気は晴れです,0
        これはペンです,1
        明後日の天気は雨です,0
        これはりんごです,1
