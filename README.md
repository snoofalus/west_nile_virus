# Kaggle: West Nile Virus
The code of this repository is an unofficial python project meant as a learning experience and not as a submitted answer to the 2015 kaggle challenge [West Nile Virus Prediction](https://www.kaggle.com/c/predict-west-nile-virus/overview).

Its objectives are twofold:
- The primary goal is leveraging feature engineering to gain further insights of the dataset, aid selection of an ML prediction algorithm and to improve the performance of said algorithm.
- The secondary goal is to perform a basic implementation of a prediction algorithm to evaluate and explain findings from the prior feature engineering.

The original Kaggle challenge provides weather-, testing- and spray data from mosquitos in the Chicago area. This can be used to predict when and where different mosquito species will test positive for West Nile Virus. Better prediction models would greatly alleviate the city of Chicago and their health departments in their efforts to stop the spread of this deadly virus.

## Prerequisites
- python 3.9.7
- numpy 1.21.2
- pandas 1.3.4
- scikit-learn 1.0.1
- matplotlib 3.5
- seaborn 0.11.2
- pytorch 1.10.0
- torchvision 0.11.1

## Usage
The model expects the following folder:  
A datafolder @ src/data

and files:  
Training dataset @ src/data/train.csv  
Test dataset @ src/data/test.csv  
GIS data of spraying in 2011 and 2013 @ src/data/spray.csv  
Weather data from 2007 to 2014 @ src/data/weather.csv


The visualization and data exploraton can then be run by
```python 
python visualize.py 
```
to show descriptions and save descriptory images from the dataset.

Afterwards run the main-file with train mode and [rforest] or [neural]
```python
python main.py --train-mode neural --predictor neural
```
to train a predictor model on the dataset and calculate training and validation accuracy.

## List of files
"main.py" 
Main file used for choosing hyperparameters, prediction model

"visualize.py"
Used for data exploration methods

"functions.py"
Helper functions used to perform random forest, evaluate accuracy, create data loggers, etc.

"architectures/networks"
Architectures used for neural net models.

"dataset/custom_datasets.py"
Custom dataloader and dataset classes for neural models

## References
```
Department of Public Health, Chicago (2015, April). 
West Nile Virus Prediction. 
Retrieved November 28, 2021 from 
https://www.kaggle.com/c/predict-west-nile-virus/data.
```
