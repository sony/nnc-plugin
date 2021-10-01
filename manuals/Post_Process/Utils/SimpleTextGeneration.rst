Utils/Simple Text Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate an English sentence using a trained model that predicts the index of the next word based on the index series x of the input string and the input string length l (lowercase L).

The output sentence is generated until the generated string reaches the maximum sentence length of the model, the model outputs the 0th category indicating EoS (End of Sentence), or the same word becomes consecutive.



.. list-table::
   :widths: 30 70
   :class: longtable

   * - model
     -
        Specify the model file (*.nnp) for text generation.
        
        To generate text based on the training result selected in the Evaluation tab, use the default results.nnp.

   * - input-variable
     - Specify the variable name that indicates the index series of the input string in the trained model.

   * - length-variable
     - Specify a variable name that indicates the number of words in the input string in the trained model.

   * - index-file-input
     -
        Specify the filename of an existing word index CSV file.
        
        A word index CSV file is a CSV file that consists of an index starting from 0 in the first column and words in the second column, and each line consists of a word index and words.

   * - seed-text
     -
        Specify the text from which the text is generated.
        
        The sentence following the string entered here will be generated.

   * - normalize
     - Specify whether to perform normalization processing that unifies the same characters with different character codes for the text entered in seed-text.

   * - mode
     -
        Specify the method of text generation.
        
        sampling: Sampling is performed based on the predicted probability of each word to determine the next word.
        
        beam search: Using beam search, determine the next word while leaving candidates with a high probability of generating the entire generated text.

   * - temperature
     -
        Specify temperature parameters when using sampling mode.
        
        The higher the value, the easier it is to select words with a low probability.
        
        The smaller the value, the easier it is to select the word with the highest probability.

   * - num-text
     - Specify the number of texts to include in the result.

   * - output
     - Specify the file name of the CSV file that outputs text generation results.


**Reference**

This plugin can be used to check the results of the 20newsgroups_lstm_language_model and 20newsgroups_transformer_language_model sample project.

Utils/Simple Japanese Text Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates a Japanese sentence using a trained model that predicts the index of the next word based on the index series x of the input string and the input string length l (lowercase L).

For how to use this plugin, please refer to the explanation about Simple Text Generation plugin.

