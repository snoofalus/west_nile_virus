#standard library
import os

#data
import numpy as np
import pandas as pd

#torch
import torch

#plotting
from pandas.plotting import table
import matplotlib.pyplot as plt

def get_df_description(dataframe, sample_index):
	'''
	Args:
		dataframe: a pandas dataframe of feature and target vars
		sample_index: 
	Returns:
		prints: dataframe description and sample info
	'''

	#summary of data 
	print(dataframe.describe(), end='\n\n')

	#columns of data frames
	print('Column names:')
	print(dataframe.columns, end='\n\n')

	#df shape in tuple of (rows, cols) = (10506, 12)
	print('Shape of train df as (rows, cols):')
	print(dataframe.shape, end='\n\n') 

	#int index out 3rd sample
	print('Data Sample:')
	print(dataframe.iloc[2], end='\n\n')

	#check if df has Nan values
	is_nan = dataframe.isnull().values.any()
	print('Dataframe contains Nan:')
	print(is_nan, end='\n\n')

	#if so number of Nan 
	n_nan = dataframe.isnull().sum().sum()
	print('Number of Nan values:')
	print(n_nan, end='\n\n')

def plot_and_save_sample(data_frame, sample_index, image_dir, image_name):
	fig = plt.figure()
	ax = plt.subplot(111, frame_on=False) # no visible frame
	ax.xaxis.set_visible(False)  # hide the x axis
	ax.yaxis.set_visible(False)  # hide the y axis

	table(ax, data_frame.iloc[2], loc='center')

	file_name = image_name
	file_path = os.path.join(image_dir, file_name)
	plt.savefig(file_path, bbox_inches='tight')

def np_get_accuracy(predictions, targets):
	'''
	Binary classification accuracy found by: 
	correct_preds / total_preds

	Args:
		predictions: predicted target variables
		targets: ground truth target variables
	Returns:
		accuracy: prediction accuracy 
	'''

	correct_predictions = (predictions == targets).sum()
	total_predictions = predictions.size
	accuracy = correct_predictions / total_predictions

	return accuracy

def pt_get_accuracy(logits, targets):
	'''
	Calculates binary accuracy average over a minibatch by first squishing raw 
	logit predictions between [0, 1] and then thresholding probabilities at 0.5
	Takes in logits.detached() and masks.detached() to leave no risk of changing computational graph.

	correct_preds / n_batch

	Args:
		logits: raw predictions from (-inf, inf)
		targets: ground truth target variables
	Returns:
		accuracy: prediction accuracy in batch
	'''
	targets = targets.int()
	probability_predictions = torch.sigmoid(logits) #probabilities
	predictions = torch.round(probability_predictions).int()

	correct_predictions_batch = torch.sum(torch.eq(predictions, targets)).item()

	total_predictions_batch = torch.numel(targets)

	avg_batch_acc =correct_predictions_batch / total_predictions_batch

	return avg_batch_acc

def save_checkpoint(state, is_best, save_dir):

	file_name = 'checkpoint.pth'#'checkpoint.{}.ckpt'.format(epoch)
	path = os.path.join(save_dir, file_name)

	torch.save(state, path)

	#if the current model has best validation acc/loss until now
	#then copy checkpoint model to a separate best model
	#(path to current file, path to best copy)
	#if is_best:
		#shutil.copyfile(path, os.path.join(save_dir, 'best_model.pth'))

#recommended from one of the pytorch devs
#https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

class TextLog(object):
	'''
	A logger object for saving accuracies and ious to a text file.

	Usage:
		1. initialize and open file with log_path.
		2. create headers for variables to save.
		3. update variables each validation run.
		4. close after last epoch.
	'''

	def __init__(self, log_path):
		
		self.file = open(log_path, 'w')

	def headers(self, headers):
		#creates headliners for log file
		#self.metrics = {}

		for head in headers:
			self.file.write(head)
			self.file.write('\t')
			#self.metrics[head] = []
		self.file.write('\n')
		self.file.flush()

	def update(self, metrics):
		#save ('Epoch', 'Total Ep', 'Train Loss', 'Val Loss', 'Val Acc')
		for metric in metrics:
			self.file.write('{}'.format(metric))
			self.file.write('\t')
		self.file.write('\n')

		self.file.flush()

	#run once after last epoch to close writefile.
	def close(self):
		self.file.close()