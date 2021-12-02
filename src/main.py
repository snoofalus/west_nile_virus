#standard library
import os
import argparse

#misc
import timeit

#data
import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble, model_selection, decomposition

#plotting
import seaborn as sns
import matplotlib.pyplot as plt

#torch
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

#local
from functions import *
from architectures import networks


parser = argparse.ArgumentParser(description='Predict West Nile Virus')

#run options
parser.add_argument('--train-mode', action='store_true',
					help='If command-line argument present train prediction model on the dataset. If argument not present show dataset descriptions and save descriptor images')

#choose prediction model
parser.add_argument('--predictor', default='neural',
					help='Prediction model, i.e. random forest or neural net: [rforest], [neural].')

#random tree hyperparams
parser.add_argument('--n-trees', default=100, type=int,
					help='Number of trees in the forest') 

#neural net optimizations
parser.add_argument('--epochs', default=40, type=int,
					help='number of epochs')

parser.add_argument('--eval-epochs', default=5, type=int,
					help='evaluate model on validation set between every eval_epochs (integer) iterations')

parser.add_argument('--batch-size', default=64, type=int, metavar='N',#32, 64, 128
					help='train batchsize')

parser.add_argument('--learning-rate', default=0.0001, type=float,
					metavar='LR', help='learning rate')

#fully connected hyperparameters
parser.add_argument('--regularization', default=5e-4, type=float,
					help='adds a penalty term to the error function to reduce overfitting')

parser.add_argument('--dropout', default=1, type=float,
					help='chance of ignoring neurons in fwrd bkwrd pass during training to reduce overfitting')

#arch datasets and pathing
parser.add_argument('--arch', default='lin32',
					help='Choose architecture from [lin32].')

parser.add_argument('--dataset', default='wnile',
					help='Choose dataset from: [wnile]')

parser.add_argument('--run-name', default='',
					help='additional text appended to savename to distinguish run by arch@dataset_<run-name>.')

args = parser.parse_args()


#use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#device_count = torch.cuda.device_count()
#device_name = torch.cuda.get_device_name(0)
#device_index=torch.cuda.current_device()

#make experiments reproducible
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

#save model state separate from checkpoint file if current avg_epoch_acc or avg_epoch_loss is best until now
best_epoch_acc = 0.0
#best_epoch_loss = 2000.0

'''
Algorithm 1
Basic NN AutoEncoder
n_feats -> (n_feats, 32) -> (32, n_target) -> out
'''

#Example run options for training
###-------------------------------------------------------------------------
#python main.py --train --predictor rforest
#python main.py --train-mode --predictor neural --run-name run1

def main():

	#Load data and create directories
	###-------------------------------------------------------------------------

	root = os.path.abspath(os.getcwd())

	train_dir = os.path.join(root, 'data/train.csv')
	test_dir = os.path.join(root, 'data/test.csv')
	weather_dir = os.path.join(root, 'data/weather.csv')
	spray_dir = os.path.join(root, 'data/spray.csv')

	df_train = pd.read_csv(train_dir)
	df_test = pd.read_csv(test_dir)
	df_weather = pd.read_csv(weather_dir)
	df_spray = pd.read_csv(spray_dir)

	#Preprocess features
	###-------------------------------------------------------------------------

	#separate feature and target variables
	x_train = df_train.drop(['WnvPresent', 'Address', 'Block', 'Street', 'AddressNumberAndStreet'], axis=1)
	y_train = df_train['WnvPresent']
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

	x_train['Species'] = le.fit_transform(x_train['Species'])

	#print(x_train['Trap'].value_counts())
	x_train['Trap'] = le.fit_transform(x_train['Trap'])
	#print(x_train['Trap'].value_counts())


	#Train prediction model
	###-------------------------------------------------------------------------
	if args.train_mode:

		#aggregate/gby Latitude and Longitude?
		#TODO


		#use .values to get values as np array from df
		x_train = x_train.values
		y_train = y_train.values 

		#save scaler from train for later use in nn
		scaler = preprocessing.StandardScaler()
		scaler.fit(x_train)


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

			#assert save directory do not exist then create one
			arch_data_name = '{}@{}_{}'.format(args.arch, args.dataset, args.run_name)
			save_dir = os.path.join('results', arch_data_name)

			print(save_dir)

			assert not os.path.exists(save_dir)
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)

			if args.dataset =='wnile':

				import dataset.custom_dataset as dataset

				num_features = 8
				num_out = 1#2 classes 1 logit

				x_train = scaler.transform(x_train)
				x_val = scaler.transform(x_val)

				train_dataset, val_dataset = dataset.get_dataset(x_train, x_val, y_train, y_val)

				train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
				val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

			#Train prediction model
			#
			##-----------------------------------------------------------------
			params = dict(vars(args))

			if args.arch =='lin32':
				print('==> creating Linear32 model')

				filter_sizes = [32]


			model = networks.Linear32(num_features=num_features, filter_sizes=filter_sizes, num_out=num_out, **params)

			model.to(device)

			criterion = nn.BCEWithLogitsLoss()#binary cross entr + sigmoid

			#could anternatively output 2 nodes and use crossentropy for binary classification
			#see: https://discuss.pytorch.org/t/two-output-nodes-for-binary-classification/58703
			#criterion = nn.CrossEntropyLoss()

			optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

			#initalize log file, open for writing, create headers
			log_name = 'log.txt'
			log_path = os.path.join(save_dir, log_name)

			text_log = TextLog(log_path)
			text_log.headers(['Epoch', 'Total Ep', 'Train Loss', 'Val Loss', 'Val Acc'])

			for epoch in range(args.epochs):

				#train one epoch on training data
				train_epoch_loss = train(train_loader, model, criterion, optimizer, epoch, args)

				#run validation on val dataset every eval_epochs
				if epoch % args.eval_epochs == 0:
					print('Evaluating model on validation set')
					val_epoch_loss, val_epoch_acc = validate(val_loader, model, criterion, epoch, args)

				is_best = val_epoch_acc > best_epoch_acc
				#is_best = val_loss < best_loss

				#to resume training or run inference, minimum needed is both model state dict and optimizer state dict
				#save checkpoints has input (dict, checkpoint directory)
				save_checkpoint({'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'acc': val_epoch_acc,
				'loss': val_epoch_loss,
				}, is_best, save_dir)

				#save to log file
				metrics = [epoch, args.epochs, train_epoch_loss, val_epoch_loss, val_epoch_acc]
				text_log.update(metrics)

			#close log file after last training
			text_log.close()

		else:
			print('ERROR No valid classifier. Please choose from [rforest], [neural]')

def train(train_loader, model, criterion, optimizer, epoch, args):

	loss_meter = AverageMeter()

	model.train()

	num_batches = len(train_loader)

	for batch_idx, (data, targets) in enumerate(train_loader):

		data, targets = data.to(device), targets.to(device)

		#forward->backpropation->loss
		logits = model(data) #logits run from (-inf, inf)
		
		#(n, ) -> (n, 1)
		targets = torch.unsqueeze(targets, 1)

		#loss of entire minibatch divided by N batch elements
		loss = criterion(logits, targets)#BCE includes sigmoid->probs

		optimizer.zero_grad()
		loss.backward()

		#update model parameters
		optimizer.step()

		batch_loss = loss.detach().item()
		loss_meter.update(batch_loss, data.shape[0])

		if batch_idx %50 == 0:
			avg_batch_acc = pt_get_accuracy(logits.detach(), targets.detach())
			print('Epoch: [{}/{}] Batch: [{}/{}] Loss: [{:.4e}] Acc: [{:.3f}]'.format(epoch, args.epochs, batch_idx, num_batches, batch_loss, avg_batch_acc))


	return loss_meter.avg

def validate(val_loader, model, criterion, epoch, args):

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()

	model.eval()

	with torch.no_grad():

		num_batches = len(val_loader)

		for batch_idx, (data, targets) in enumerate(val_loader):

			data, targets = data.to(device), targets.to(device)

			logits = model(data)

			#(n, ) -> (n, 1)
			targets = torch.unsqueeze(targets, 1)

			#loss of entire minibatch divided by N batch elements
			loss = criterion(logits, targets)
			batch_loss = loss.detach().item()
			loss_meter.update(batch_loss, data.shape[0])

			batch_acc = pt_get_accuracy(logits.detach(), targets.detach())
			acc_meter.update(batch_acc, targets.shape[0])

		#print avg val loss and accuracy for one epoch
		print('Epoch: [{}/{}] Avg Epoch Val Loss:[{:.4e}] Avg Epoch Val Acc: [{:.3f}]'.format(epoch, args.epochs, loss_meter.avg, acc_meter.avg))

		#val_epoch_loss, val_epoch_acc
		return loss_meter.avg, acc_meter.avg

main()






