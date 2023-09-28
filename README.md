# Customer Conversion Prediction

## Overview

Predict the likelihood of customer conversion based on customer behavior using machine learning. This project utilizes the "Customer Propensity to Purchase" dataset from Kaggle and employs various machine learning models to make predictions. To handle the imbalanced training data, under-sampling is applied. The final prediction is generated through a "voting" classifier, resulting in a high accuracy rate exceeding 99%.

## Project Structure

The project structure is organized as follows:

- `src/`: Contains the source code for the project.
- `data/`: Directory for storing data files.
- `results/`: Directory for storing prediction results.
- `README.md`: This documentation file.

## Requirements

Make sure you have the following requirements installed:

- Python (>=3.6)
- Libraries listed in `requirements.txt`

You can install the required libraries using pip:

```
pip install -r requirements.txt
```

## Usage
Place your input data file (e.g., input.csv) in the data/ directory.

Run the `process.py` script with the input and output file paths as arguments:
`python process.py data/input.csv results/results.csv`
The predictions will be saved in the specified output file (`results/results.csv`).

## Model Details
This project employs several "weak" classifiers, including Logistic Regression, Ridge Classifier, SGD Classifier, K-Nearest Neighbors, Support Vector Classifier, Random Forest Classifier, and Gaussian Naive Bayes. The predictions from these classifiers are combined using a "voting" classifier, resulting in the final prediction.

## Results
The model achieves an accuracy rate exceeding 99%, making it highly effective at predicting customer conversions.
