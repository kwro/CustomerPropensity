## Customer Propensity Classifier

&nbsp;
### Description
Predict the likelihood of customer conversion based on customer behaviour.

&nbsp;
### Usage
You can use the pre-trained model or upload your file to train a model. There is huge flexibility in terms of input data- you are free to process features of any number and any kind. The input file should contain a csv file with one column called "uri". There should be one row containing a dictionary pointing the path to the train data and data to perform predictions on. 

&nbsp;
### Background
The actual model is pre-trained on Kaggle "Customer Propensity to Purchase" dataset. It takes into account the imbalanced train set and performs under-sampling. There are several different "weak" classifiers, which predictions are then processed by "voting" classifier. This process creates a synergy effect with an accuracy exceeding 99%. 
