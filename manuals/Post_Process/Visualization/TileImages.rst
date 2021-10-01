Visualization/Tile Images
~~~~~~~~~~~~~~~~~~~~~~~~~

This plugin creates an image by tiling the images included in the input CSV file. This can be used, for example, to list incorrectly classified images.



.. list-table::
   :widths: 30 70
   :class: longtable

   * - input
     -
        Specify the dataset CSV file containing the image file names.
        
        To create an image by tiling the images contained in Output Result of the Evaluation tab, use the default output_result.csv.
        
        To perform processing based on results obtained by filtering Output Result (e.g., incorrectly classified images), save the filtered results to a CSV file using the shortcut menu, and specify the name of the saved file in input.

   * - variable
     -
        Of the variables included in the dataset CSV file specified by input, specify the variable name that includes the images to be tiled.
        
        If this is not specified, all images for which file names are included in the CSV file specified by input will be processed.

   * - image_width
     -
        Specify the width of a single input image. If this is not specified, the width of the first image is used.
        
        If the size of an input image is different from the size specified by image_width, the width of the image will be resized to the value specified by image_width.

   * - image_height
     -
        Specify the height of a single input image. If this is not specified, the height of the first image is used.
        
        If the size of an input image is different from the size specified by image_height, the width of the image will be resized to the value specified by image_height.

   * - num_column
     -
        Specify the number of columns to tile the images.
        
        The maximum horizontal width of the output image is image_width num_column.

   * - start_index
     - Specify the index of the first data sample to display in tiles.

   * - end_index
     -
        Specify the index of the last data sample to display in tiles.
        
        end_index start_index + 1 pages of images are displayed in tiles.
        
        If end_index is not specified, all images contained in all the data in the CSV file specified by input will be displayed.

   * - Output
     -
        Specify the name of the image file to output the tiled images to.
        
        If Tile Images is executed from the evaluation results of the Evaluation tab, the tiled images are saved to the specified file in the training result folder.


