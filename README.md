# Plugins for Neural Network Console \(English/[日本語](README_ja.md)\)

![](./img/plugin.png)

## What is the plugins for Neural Network Console ?
The plugins enable pre-processing and post-processing on Neural Network Console. There are already plugins on Neural Network Console, but you can use the latest release of the pulings from this repository.


## Current lineup of the plugins
###  Pre-processing
* Create dataset
    * [Object detection (from Yolo v2 format)](./manuals/Pre_Process/Create_Dataset/ObjectDetection.rst)
    * [Object detection (for CenterNet from Yolo v2 format)](./manuals/Pre_Process/Create_Dataset/ObjectDetection_CenterNet.rst)
    * [Simple  (Japanese) text classification](./manuals/Pre_Process/Create_Dataset/SimpleTextClassification.rst)
    * [String classification](./manuals/Pre_Process/Create_Dataset/StringClassification.rst)
    * [Simple tabular data](./manuals/Pre_Process/Create_Dataset/SimpleTabularDataset.rst)
    * [Split image](./manuals/Pre_Process/Create_Dataset/SplitImage.rst)
    * [Split wav](./manuals/Pre_Process/Create_Dataset/SplitWav.rst)

###  Post-processing
* Utils
    * [Cross tabulation](./manuals/Post_Process/Utils/CrossTabulation.rst)
    * [CSV to wav](./manuals/Post_Process/Utils/CSVtoWAV.rst)
    * [Inference](./manuals/Post_Process/Utils/Inference.rst)
    * [Parameter stats](./manuals/Post_Process/Utils/ParameterStats.rst)
    * [Restore split images / wav](./manuals/Post_Process/Utils/RestoreSplitImageWav.rst)
    * [Similar words](./manuals/Post_Process/Utils/SimilarWords.rst)
    * [Simple (Japanese) text generation](./manuals/Post_Process/Utils/SimpleTextGeneration.rst)
    * [tSNE](./manuals/Post_Process/Utils/tSNE.rst)
    
* Visualization
    * [Scatter plot](./manuals/Post_Process/Visualization/ScatterPlot.rst)
    * [Tile images](./manuals/Post_Process/Visualization/TileImages.rst)

* eXplainable AI (XAI)
    * [SGD influence](./manuals/Post_Process/XAI/SGDInfluence.rst)
    * [Influence Functions](./manuals/Post_Process/XAI/InfluenceFunctions.rst)
    * [Face evaluation](./manuals/Post_Process/XAI/FaceEvaluation.rst)
    * [Grad-CAM](./manuals/Post_Process/XAI/GradCAM.rst)
    * [LIME](./manuals/Post_Process/XAI/LIME.rst)
    * [SHAP](./manuals/Post_Process/XAI/SHAP.rst)
    * [Smooth Grad](./manuals/Post_Process/XAI/SmoothGrad.rst)
    * [TracIn](./manuals/Post_Process/XAI/TracIn.rst)

## How to use the latest plugins
The plugins run on Neural Network Console. If you do not have Neural Network Console, please download from here (https://dl.sony.com/).
1. Download the zip files from this repository. 
2. Extract the zip files on your PC.
3. Delete the existing plugins folder. You can find it from neural_network_console > libs > **plugins**. 
* **NOTE** If you do not want to turn off some plugins, please leave them.

4. Put the downloaded **plugins** folders in the same place, neural_network_console > libs > **plugins**.  

###  Pre-processing
* To execute the plugins of the pre-processing, select the "DATASET" on the left of the top screen. Then  click "Create Dataset", you can select the plugins of the pre-processing.
<p align="center">
<img src="./img/Preprocessing.png" width="400px">  
</p>


### Post-processing
* To execute the plugins of the post-processing, right-click the evaluation results on the Evaluation tab to open a shortcut menu and select the plugins.
<p align="center">
<img src="./img/postprocessing.png" width="400px">  
</p>


