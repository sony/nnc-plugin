Utils/t-SNE
~~~~~~~~~~~

This plugin executes t-SNE, which is a method used to compress the dimension of high-dimensional data.

Visualizing Data using t-SNE
   - \L. van der Maaten, G. Hinton.
   - http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

The compression results using t-SNE are output to a CSV file specified in output. To visualize the results of t-SNE as a scatter plot, execute the t-SNE plugin, then execute the Scatter Plot plugin with the CSV file specified in t-SNE’s output specified for input, and the variable names (x_tsne__0, x_tsne__1, and so on) of the t-SNE results specified for x and y.



.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     -
        Specify the dataset CSV file to be used in the t-SNE computation.
        
        To perform t-SNE based on the Output Result shown in the Evaluation tab, use the default output_result.csv.

   * - variable
     - Of the variables included in the dataset CSV file specified in input, specify the name of the variable to be used in the t-SNE computation.

   * - dim
     - Specify the number of dimensions to compress the data to.

   * - output
     - Specify the name of the CSV file to output the t-SNE results to.


