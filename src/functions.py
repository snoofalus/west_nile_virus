#misc
import numpy as np


def np_get_accuracy(y_hat, y):
	'''
	Args:
		y_hat: predicted target variables
		y: ground truth target variables
	Output:
		accuracy: prediction accuracy found by dividing correct_preds / n_preds
	'''

	correct_preds = (y_hat == y).sum()
	n_preds = y_hat.size
	accuracy = correct_preds / n_preds

	return accuracy