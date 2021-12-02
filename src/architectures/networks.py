#torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear32(nn.Module):
	def __init__(self, num_features, filter_sizes, num_out, **kwargs):
		super(Linear32, self).__init__()

		'''
		Basic fully connected implementation with linear layers.
		Args:
			num_features: 
			filters_sizes:
			fully_connected_sizes:
		'''

		self.activation = nn.ReLU()

		self.linear1 = nn.Linear(num_features, filter_sizes[0])
		self.batchnorm1 = nn.BatchNorm1d(filter_sizes[0]) 
		self.dropout1 = nn.Dropout(0.5) 

		self.linear2 = nn.Linear(filter_sizes[0], filter_sizes[0])
		self.batchnorm2 = nn.BatchNorm1d(filter_sizes[0]) 
		self.dropout2 = nn.Dropout(0.5)

		self.fully_connected = nn.Linear(filter_sizes[0], num_out)

	def forward(self, x):

		x = self.activation(self.batchnorm1(self.linear1(x)))
		x = self.dropout1(x)

		x = self.activation(self.batchnorm2(self.linear2(x)))
		x = self.dropout1(x)

		x = self.fully_connected(x)

		return x