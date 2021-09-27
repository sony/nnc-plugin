# SHAP (Image)
Using a method called SHAP, the areas of the input image that affect the classification result are made visible in the model, which performs image classification. It is shown in red over the original image for the positively affected area in the classification, while in blue for the negatively affected area.

>A Unified Approach to Interpreting Model Predictions
Scott Lundberg, Su-In Lee
https://arxiv.org/abs/1705.07874


# SHAP (Image batch)
Using a method called SHAP,  the areas of the input image that affect the classification result are made visible in the model, which performs image classification. SHAP(batch) processes all images in the specified dataset, while SHAP processes a single image.


# Kernel SHAP (Tabular)
Using a method called Kernel SHAP, a classification result is explained with the contribution of the features in input table data. Each feature is explained with degree of contribution, which enables to interpret the classifier judgement.

>A Unified Approach to Interpreting Model Predictions
Scott Lundberg, Su-In Lee
https://arxiv.org/abs/1705.07874

Property | Notes
-- | --
model | Specify the model file (*.nnp) that will be used in the Kernel SHAP computation. To perform Kernel SHAP based on the training result selected in the Evaluation tab, use the default results.nnp.
input | Specify the dataset CSV file containing the data to analyze.
train | Specify the dataset CSV file used for the training of the model of interest.
index | Specify the index of the data in the input CSV.
alpha | Specify the coefficient for the regularization term of Ridge regression.
class_index | Specify the index of the class of the data to analyze. Default value is 0. For regression model or binary classification model, only class_index=0 can be specified.
output | Specify the name of the CSV file to output the inference results to.


# Kernel SHAP (Tabular Batch)
Using a method called Kernel SHAP, a classification result is explained with the contribution of the features in input table data. Each feature is explained with degree of contribution, which enables to interpret the classifier judgement. Kernel SHAP(tabular batch) processes all records in the specified dataset, while Kernel SHAP(tabular) processes a single record.

Property | Notes
-- | --
model | Specify the model file (*.nnp) that will be used in the Kernel SHAP computation. To perform Kernel SHAP based on the training result selected in the Evaluation tab, use the default results.nnp.
input | Specify the dataset CSV file containing the data to analyze.
train | Specify the dataset CSV file used for the training of the model of interest.
class_index | Specify the index of the class of the data to analyze. Default value is 0. For regression model or binary classification model, only class_index=0 can be specified.
alpha | Specify the coefficient for the regularization term of Ridge regression.
output | Specify the name of the CSV file to output the inference results to.
