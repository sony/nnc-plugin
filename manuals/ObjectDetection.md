# Object Detection (Yolo v2 format)
This plugin converts the object detection dataset prepared in Yolo v2 format to Neural Network Console dataset CSV file format, especially to the format used in object detection sample project “synthetic_image_object_detection”.
For Yolo v2 format, prepare a text file with the same name as the image file and a file name with the extension changed to “.txt”, and enter the index, center X coordinate, center Y coordinate, width, and height of the object in each line of the text file. The index of the object is an integer starting from 0, and the coordinates and vertical and horizontal sizes are specified with (0.0, 0.0) at the upper left of the image and (1.0, 1.0) at the lower right of the image.

## Reference
The object detection dataset created using this plugin and with the grid-size set to 16 can be trained by using the synthetic_image_object_detection sample project.
For training, for each Argument layer in the Training, Validation, and Runtime networks, set the Value property of NumClass to the value specified by num-class, set NumAnchor to the value specified by anchor, set InputCh to the value specified by channel, and set Width and Height to the values specified by width and height, respectively.
