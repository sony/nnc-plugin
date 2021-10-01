Object Detection (Yolo v2 format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This plugin converts the object detection dataset prepared in Yolo v2 format to Neural Network Console dataset CSV file format, especially to the format used in object detection sample project “synthetic_image_object_detection”.

For Yolo v2 format, prepare a text file with the same name as the image file and a file name with the extension changed to “.txt”, and enter the index, center X coordinate, center Y coordinate, width, and height of the object in each line of the text file. The index of the object is an integer starting from 0, and the coordinates and vertical and horizontal sizes are specified with (0.0, 0.0) at the upper left of the image and (1.0, 1.0) at the lower right of the image.

image.png

image.txt

.. list-table::
   :widths: 100

.. code::
   
    1 0.9764947367828946 0.6157956367918462 0.3083005211565641 0.16941592385576362
    0 0.5255774544381083 0.7251694243924344 0.1763326673913164 0.13065036222690296
    3 0.8369506590907345 0.9836015569071715 0.2502855465179791 0.19769644444902154
    2 0.48402106995254446 0.5825400718280271 0.32631333545360164 0.23807745759339757
    …

.. list-table::
   :widths: 30 70
   :class: longtable

   * - input-dir
     - Specify the folder that contains the images and text files of the Yolo v2 format source dataset.

   * - output-dir
     - Specify the folder to output the created dataset to.

   * - num-class
     - Specify the number of object types to be detected in the source dataset (maximum object index +1)

   * - channel
     - Set the number of color channels in the output images to 1 (monochrome) or 3 (RGB color).

   * - width
     - Specify the output image width in pixels.

   * - height
     - Specify the output image height in pixels.

   * - anchor
     - Specify the number of anchors (the number of variations of aspect ratio and size that are the basis for object detection).

   * - grid-size
     -
        Specify the size of the grid that makes up the image specified by width and height in pixels.
        
        width and height must be divisible by grid size.

   * - mode
     -
        If the aspect ratio of the images in each folder is different from the aspect ratio indicated by the specified width and height, specify the method for aligning the aspect ratio.

        |
        
        trimming: The edges are trimmed to align the aspect ratio.
        
        padding: The edges are padded with zeros to align the aspect ratio.
        
        resize: The image is resized to the specified size, ignoring the aspect ratio.

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

The object detection dataset created using this plugin and with the grid-size set to 16 can be trained by using the synthetic_image_object_detection sample project.

For training, for each Argument layer in the Training, Validation, and Runtime networks, set the Value property of NumClass to the value specified by num-class, set NumAnchor to the value specified by anchor, set InputCh to the value specified by channel, and set Width and Height to the values specified by width and height, respectively.

