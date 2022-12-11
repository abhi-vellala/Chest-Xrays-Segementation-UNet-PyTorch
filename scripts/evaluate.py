import numpy as np
import pandas as pd
from IPython.display import display
# import distances as distances
import matplotlib.pyplot as plt

from scipy import spatial

def euclidean(A, B):
    return np.sqrt(np.sum(np.square(A - B)))

def manhattan(A, B):
    return np.sum(np.abs(A - B))

def cosine(A, B):
    dot_prod = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_prod/(norm_a * norm_b)

class EvaluateImageSegmentation:

	def __init__(self, groundtruth_mask, predicted_mask):
		self.gt_mask = groundtruth_mask
		self.pred_mask = predicted_mask
		self.groundtruth_mask = groundtruth_mask.astype(bool)
		self.predicted_mask = predicted_mask.astype(bool)
		self.intersection = self.groundtruth_mask * self.predicted_mask
		self.union = self.groundtruth_mask + self.predicted_mask
		self.true_positive = self.intersection
		self.false_positive = self.union != self.groundtruth_mask
		self.false_negative = self.union != self.predicted_mask
		self.true_negative = np.invert(self.union)

	def accuracy(self):
		return 1 - np.sum(self.true_positive+self.true_negative)/np.sum(self.true_positive+self.true_negative+self.false_negative+self.false_positive)

	def precision(self):
		return np.sum(self.true_positive)/np.sum(self.true_positive+self.false_positive)

	def recall(self):
		return np.sum(self.true_positive)/np.sum(self.true_positive+self.false_negative)

	def f1score(self):
		return 2*np.sum(self.true_positive)/(2*np.sum(self.true_positive)+np.sum(self.false_positive)+np.sum(self.false_negative))

	def dice(self):
		return 2*np.sum(self.true_positive)/(2*np.sum(self.true_positive)+np.sum(self.false_positive)+np.sum(self.false_negative))

	def IoU(self):
		return np.sum(self.intersection)/np.sum(self.union)

	def get_confusion_matrix(self):
		gt_series = pd.Series(self.groundtruth_mask.flatten(), name="ground truth")
		pred_series = pd.Series(self.predicted_mask.flatten(), name="predicted")
		df_confusion = pd.crosstab(gt_series, pred_series)
		display(df_confusion)

	def hausdorff_distance(self, distance='euclidean'):
		n1 = self.gt_mask.shape[0]
		n2 = self.pred_mask.shape[0]
		cmax = 0
		for i in range(n1):
			cmin = np.inf
			for j in range(n2):
				# dist = getattr(distances, distance)
				dist_cal = euclidean(self.gt_mask[i,:], self.pred_mask[j,:])
				dist = np.sqrt(np.sum(np.square(self.gt_mask[i,:] - self.pred_mask[j,:])))
				if dist_cal < cmin:
					cmin = dist_cal
				if cmin < cmax:
					break
			if cmin > cmax and np.inf > cmin:
				cmax = cmin
		return cmax


def evaluate_helper(loss_dict):
    loss_dict_modified = {}
    for key in loss_dict.keys():
        if key == "hausdorf":
            loss_dict_modified[key] = sum(loss_dict[key])/len(loss_dict[key])
        else:
            loss_dict_modified[key] = sum(loss_dict[key])/len(loss_dict[key])
        
    return loss_dict_modified
    

# def plot_evalution_metrics(loss_dict):
    
#     fig.axs = plt.subplot((3,2), figsize=(15,8))
    