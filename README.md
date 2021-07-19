# Plugins of Neural Network Console

![](./img/plugin.png)

## What is the plugins of Neural Network Console ?
The plugins enable pre-processing and post-processing on Neural Network Console. There are already plugins on Neural Network Console, but you can use the latest release of the pulings from this repository.


## Current lineup of the plugins
###  Pre-processing
* Create dataset
    * [Object detection](./img/ObjectDetection.md)
    * [Simple (Japanese) text classification](./img/SimpleTextClassification.md)
    * [String classification](./img/StringClassification.md)
    * [Simple tabular data](./img/SimpleTabularDataset.md)
    * [Split image](./img/SplitImage.md)
    * [Split wav](./img/SplitWav.md)
    
###  Post-processing
* Utils
    * [Cross tabulation](./img/CrossTabulation.md)
    * [CSV to wav](./img/CSVtoWav.md)
    * [Inference](./img/Inference.md)
    * [Parameter stats](./img/ParameterStats.md)
    * [Restore split images / wav](./img/RestoreSplitWav.md)
    * [Similar words](./img/Similar_Words.md)
    * [Simple (Japanese) text generation](./img/SimpleTextGeneration.md)
    * [t-SNE](./img/tSNE.md)
    
* Visualization
    * [Scatter plot](./img/ScatterPlot.md)
    * [Tile images](./img/TileImages.md)

* eXplainable AI (XAI)
    * [SGD influence](./img/SGDinfl.md)
    * [Face evaluation](./img/FaceEvaluation.md)
    * [Grad-CAM](./img/Grad-CAM.md)
    * [LIME](./img/LIME.md)
    * [SHAP](./img/SHAP.md)
    * [Smooth Grad](./img/SmoothGrad.md)
    * [TracIn](./img/TracIn.md)

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

