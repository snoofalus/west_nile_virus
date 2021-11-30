#standard library
import os
import argparse

#misc
import timeit

#data
import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble, model_selection

#plotting
from pandas.plotting import table
import seaborn as sns
import matplotlib.pyplot as plt

#local
from functions import *

parser = argparse.ArgumentParser(description='Predict West Nile Virus')

#run options
parser.add_argument('--train-mode', action='store_true',
					help='If command-line argument present train prediction model on the dataset. If argument not present show dataset descriptions and save descriptor images')

#datasets and pathing
parser.add_argument('--dataset', default='wnile',
					help='Choose dataset from: [wnile]')

#choose prediction model
parser.add_argument('--predictor', default='neural',
					help='Prediction model, i.e. random forest or neural net: [rforest], [neural].')

#random tree hyperparams
parser.add_argument('--n-trees', default=100, type=int, metavar='N',
					help='Number of trees in the forest') 

#neural net hyperparams
parser.add_argument('--learning-rate', default=0.1, type=float,
					metavar='LR', help='learning rate')

args = parser.parse_args()

'''
Algorithm 1
Basic NN AutoEncoder
n_feats -> (n_feats, 32) -> (32, n_target) -> out
'''

def main():

	#Load data and create directories
	###-------------------------------------------------------------------------

	root = os.path.abspath(os.getcwd())

	train_dir = os.path.join(root, 'data/train.csv')
	test_dir = os.path.join(root, 'data/test.csv')
	weather_dir = os.path.join(root, 'data/weather.csv')
	spray_dir = os.path.join(root, 'data/spray.csv')

	train = pd.read_csv(train_dir)
	test = pd.read_csv(test_dir)
	weather = pd.read_csv(weather_dir)
	spray = pd.read_csv(spray_dir)

	#Explore dataframe
	###-------------------------------------------------------------------------
	
	if not args.train_mode:

		print('==> SHOWING DESCRIPTORY INFORMATION FOR THE [{}] DATASET\n'.format(args.dataset))

		#check for and create out dir
		image_dir = os.path.join(root, 'images')
		if not os.path.exists(image_dir):
			os.makedirs(image_dir)

		#summary of data 
		print(train.describe(), end='\n\n')

		#columns in data frame
		print('Column names:')
		print(train.columns, end='\n\n')

		#df shape in tuple of (rows, cols) = (10506, 12)
		print('Shape of dataframe as (rows, cols):')
		print(train.shape, end='\n\n') 

		#index out 3rd sample
		print('Data Sample:')
		print(train.iloc[2], end='\n\n')

		#slice first 3 samples
		#print(train[:3])

		#print(train['Address'][0])

		#check if df has Nan values
		is_nan = train.isnull().values.any()
		print('Dataframe contains Nan:')
		print(is_nan, end='\n\n')

		#if so number of Nan 
		n_nan = train.isnull().sum().sum()
		print('Number of Nan values:')
		print(n_nan, end='\n\n')

		#Create descriptory images
		#
		###-------------------------------------------------------------------------

		#TODO
		#histogram

		#correlation heatmap
		#TODO create correlation heatmap after encoding categorical variables
		correlation_matrix = train.corr()

		fig = plt.figure()
		sns.heatmap(correlation_matrix, vmax=0.8, annot=True, square=True)
		plt.gcf().subplots_adjust(bottom=0.15)#adjust for long column names
		file_name = 'heatmap.png'
		file_path = os.path.join(image_dir, file_name)
		plt.savefig(file_path, bbox_inches='tight')

		fig = plt.figure()
		ax = plt.subplot(111, frame_on=False) # no visible frame
		ax.xaxis.set_visible(False)  # hide the x axis
		ax.yaxis.set_visible(False)  # hide the y axis

		table(ax, train.iloc[2], loc='center')

		file_name = 'header.png'
		file_path = os.path.join(image_dir, file_name)
		plt.savefig(file_path, bbox_inches='tight')
		plt.show()

	#Train prediction model
	###-------------------------------------------------------------------------
	if args.train_mode:

		#separate feature and target variables
		x_train = train.drop(['WnvPresent', 'Address', 'Block', 'Street', 'AddressNumberAndStreet'], axis=1)
		y_train = train['WnvPresent']
		#unique, counts = np.unique(y_train, return_counts=True)
		#test = dict(zip(unique, counts))


		#parse out day and month from df
		x_train['month'] = pd.DatetimeIndex(x_train['Date']).month
		#print(x_train['month'][-5:-1])

		x_train['day'] = pd.DatetimeIndex(x_train['Date']).day
		#print(x_train['day'][-5:-1])

		x_train = x_train.drop(['Date'], axis=1)
		#print(x_train.columns)


		#encode species into numerical variable (could also use OneHotEncoder)
		#print(x_train['Species'].value_counts())

		le = preprocessing.LabelEncoder()
		#print(x_train['Species'].iloc[500])

		x_train['Species'] = le.fit_transform(x_train['Species'])
		#print(x_train['Species'].iloc[500])


		#repeat for trap labels
		#print(x_train['Trap'].value_counts())
		x_train['Trap'] = le.fit_transform(x_train['Trap'])
		#print(x_train['Trap'].value_counts())

		
		#aggregate/gby Latitude and Longitude?
		#TODO


		#use .values to get values as np array from df
		x_train = x_train.values
		y_train = y_train.values 

		#split set into train = train + val 
		x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=42)


		print('==> TRAINING [{}] PREDICTION MODEL'.format(args.predictor))

		if args.predictor == 'rforest':

			#rf is popular on kaggle, can be used as benchmark for other models
			print('==> TRAINING [{}] PREDICTION MODEL'.format(args.predictor))
			print('Train: [{}] Val: [{}]\n'.format(len(y_train), len(y_val)))

			start = timeit.default_timer()

			#normalization/scaling is not requirement for conv in rf
			clf = ensemble.RandomForestClassifier(n_estimators=args.n_trees)

			clf.fit(x_train, y_train)

			stop = timeit.default_timer()
			run_time = stop - start

			print('Completed training in {:.2f} seconds'.format(run_time))  

			y_hat_train = clf.predict(x_train)
			train_acc = np_get_accuracy(y_hat_train, y_train)

			y_hat_val = clf.predict(x_val)
			val_acc = np_get_accuracy(y_hat_val, y_val)

			print('Train Acc: [{:.3f}] Val Acc: [{:.3f}]'.format(train_acc, val_acc))

		elif args.predictor == 'neural':



			print('uwu')

		else:
			print('ERROR No valid classifier. Please choose from [rforest], [neural]')


main()






