# Plugins of Neural Network Console

![](./img/plugin.png)

## What is the plugins of Neural Network Console ?
The plugins enable pre-processing and post-processing on Neural Network Console. There are already plugins on Neural Network Console, but you can use the latest release of the pulings from this repository.


## Current lineup of the plugins
###  Pre-processing
* Create dataset
    * [Object detection](./manuals/ObjectDetection.md)
    * [Simple (Japanese) text classification](./manuals/SimpleTextClassification.md)
    * [String classification](./manuals/StringClassification.md)
    * [Simple tabular data](./manuals/SimpleTabularDataset.md)
    * [Split image](./manuals/SplitImage.md)
    * [Split wav](./manuals/SplitWav.md)
    
###  Post-processing
* Utils
    * [Cross tabulation](./manuals/CrossTabulation.md)
    * [CSV to wav](./manuals/CSVtoWav.md)
    * [Inference](./manuals/Inference.md)
    * [Parameter stats](./manuals/ParameterStats.md)
    * [Restore split images / wav](./manuals/RestoreSplitWav.md)
    * [Similar words](./manuals/Similar_Words.md)
    * [Simple (Japanese) text generation](./manuals/SimpleTextGeneration.md)
    * [t-SNE](./manuals/tSNE.md)
    
* Visualization
    * [Scatter plot](./manuals/ScatterPlot.md)
    * [Tile images](./manuals/TileImages.md)

* eXplainable AI (XAI)
    * [SGD influence](./manuals/SGDinfl.md)
    * [Influence Functions](./manuals/InfluenceFunctions.md)
    * [Face evaluation](./manuals/FaceEvaluation.md)
    * [Grad-CAM](./manuals/Grad-CAM.md)
    * [LIME](./manuals/LIME.md)
    * [SHAP](./manuals/SHAP.md)
    * [Smooth Grad](./manuals/SmoothGrad.md)
    * [TracIn](./manuals/TracIn.md)

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


