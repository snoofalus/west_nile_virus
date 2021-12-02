import os

#data
import numpy as np

#torch
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ToTensor(object):
	"""Transform np array to tensor.
	"""
	def __call__(self, x):
		x = torch.from_numpy(x)
		return x

def get_dataset(x_train, x_val, y_train, y_val):

	#if needed later can add normscale, translation/rotation transforms, etc. to compose below
	x_transform = transforms.Compose([
	ToTensor()])

	train_dataset = CustomDataset(x_train, y_train, x_transform)

	val_dataset = CustomDataset(x_val, y_val, x_transform)

	return train_dataset, val_dataset

class CustomDataset(Dataset):
	'''
	Your custom dataset should inherit Dataset and override the following methods:

		__len__ so that len(dataset) returns the size of the dataset.
		__getitem__ to support the indexing such that dataset[i] can be used to get i'th sample
	'''
	def __init__(self, data, targets, data_transform=None):
		'''
		Args:
			data(samples, features): feature data
			targets(samples, ):  0 or 1 binary labels
			transform: add transformations (e.g. to tensor, translation in images, rotation, gaussian noise)
		'''

		self.data = data
		self.targets = targets
		self.data_transform = data_transform

	def __getitem__(self, index: int):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (data, target)
		"""

		data = self.data[index, :]
		target = self.targets[index]

		if self.data_transform is not None:
			data = self.data_transform(data.astype('float32'))
		target = torch.from_numpy(np.asarray(target))

		return data, target.float()

	def __len__(self):
		return len(self.targets)