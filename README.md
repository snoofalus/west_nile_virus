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


The model can then be run with default arguments as
```python
python main.py 
```
to show descriptions and save description images from the dataset.

Alternatively add train mode with [rforest] or [neural]
```python
python main.py --train-mode neural
```
to train a predictor model on the dataset and calculate training and validation accuracy.

Try out different datasets and change hyperparameters in the main-file arg parser or through cmd-line as shown below 
```python
python train.py --train-mode --learning-rate 0.01 --dataset wnile
```

##List of files
"main.py" 
Main file used for choosing hyperparameters, prediction model and run feature engineering

"functions.py"
Functions used to perform random forest, evaluate accuracy and other metrics 

"dataset/custom_datasets.py"
Two custom pytorch dataset objects for loading images and masks.

"dataset/_dataset_.py"
Several files where _dataset_ is replaced with the name of specific datasets used for calling the dataloader objects and potentially use normalization and/or image transforms. Add your own following existing layout to use it on your own datasets. 

## References
```
Department of Public Health, Chicago (2015, April). 
West Nile Virus Prediction. 
Retrieved November 28, 2021 from 
https://www.kaggle.com/c/predict-west-nile-virus/data.
```
