Simple Tabular Dataset
~~~~~~~~~~~~~~~~~~~~~~

Converts a CSV file of a tabular dataset consisting of the header in the first row and one data per row in the second and subsequent rows into the dataset CSV format in Neural Network Console.

In the first row of tabular data, enter the name of each column.

In each cell of tabular data, enter a character string for category attributes and a number for numeric attributes. The following is an example of a CSV file for a tabular dataset with address and gender as categorical attributes and age as numeric attributes.



.. list-table::
   :widths: 100

   * -
        Address,Age,Gender
        
        Tokyo,40,Male
        
        Osaka,23,Others
        
        Kanagawa,65,Female


This plugin converts multiple explanatory variables contained in tabular data into one x vector. The categorical attribute explanatory variables are added as one hot vectors, and the numeric attribute explanatory variables are added to the vector elements as they are. Numerical attributes can be standardized by setting.



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

   * - comment-cols
     - Of the names of each column in the original CSV file, specify the name of the column that will be left as a comment in the converted dataset CSV file. To specify multiple columns, specify multiple names separated by commas.

   * - include-variables
     -
        Of the names of each column in the original CSV file, specify the names of the columns to be used as explanatory variables, separated by commas.
        
        If nothing is specified, all columns except those specified by exclude-variables and objective-variables will be used as explanatory variables.

   * - exclude-variables
     - If you do not specify include-variables, specify the names of the columns in the original CSV file that are not used as explanatory variables, separated by commas.

   * - objective-variables
     - Specifiy the name of each column in the original CSV file to use as the objective variable.
     
   * - sensitive-variables
     -
        Specifiy the name of each column in the original CSV file to use as the sensitive/protected variable (an attribute that partitions a population into groups with parity) , separated by commas.
        
        If nothing is specified, all columns except those specified by exclude-variables and objective-variables will be used as explanatory variables.

   * - standardize
     - Specify whether to standardize the value of the numeric attribute of the original CSV file to mean 0 variance 1.

   * - process-param-input
     -
        Specify the filename of an existing preprocessing parameter file.
        
        In the preprocessing parameter file, each row is each column of the original CSV file, the first column is the name, the second column is the attribute type represented by the category attribute (category) or numeric attribute (numeric), and for category attributes, the third and subsequent columns are the category names, for numeric attributes, the third column is the average, and the fourth column is the variance.
        
        .. image:: figures/simple_tabular_csv_example.png
        
        If process-param-input is specified, the input data will be converted according to the settings specified here instead of preprocessing based on the input CSV file.

   * - process-param-output
     - Specify the file name of the preprocessing parameter file to save the preprocessing parameter created based on the input CSV file.

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


