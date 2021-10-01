Utils/Similar Words
~~~~~~~~~~~~~~~~~~~

Find similar words based on the embed parameters of the words contained in the trained model.



.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Specify the model file (*.nnp) for similar word search.
        
        To search for similar words based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - parameter
     - Specify the name of the parameter used for similar word search among the parameters included in the trained model.

   * - index-file-input
     -
        Specify the filename of an existing word index CSV file.
        
        A word index CSV file is a CSV file that consists of an index starting from 0 in the first column and words in the second column, and each line consists of a word index and words.

   * - source-word
     - Specify the word that is the source of the similar word search.

   * - num-words
     - Specify the number of similar words to include in the result.

   * - output
     - Specify the file name of the CSV file that outputs similar word search results.


**Reference**

This plugin can be used to check the results of the 20newsgroups_word_embedding sample project.

