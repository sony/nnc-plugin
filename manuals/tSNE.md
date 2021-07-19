# t-SNE
This plugin executes t-SNE, which is a method used to compress the dimension of high-dimensional data.

>Visualizing Data using t-SNE
L. van der Maaten, G. Hinton.
http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

The compression results using t-SNE are output to a CSV file specified in output. To visualize the results of t-SNE as a scatter plot, execute the t-SNE plugin, then execute the Scatter Plot plugin with the CSV file specified in t-SNE’s output specified for input, and the variable names (x_tsne__0, x_tsne__1, and so on) of the t-SNE results specified for x and y.
