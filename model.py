# Import required libraries
import numpy as np  # used for array operations
from sklearn.metrics.pairwise import rbf_kernel  # used to compute Radial basis function kernel
import matplotlib.pyplot as plt  # used for plotting data
import random  # used to select random points
from matplotlib.colors import ListedColormap  # used to map fixed colors to labels
import time  # used to time operations

# Define the semi-supervised model class
class SemisupervisedModel:
    # The class initializer
    def __init__(self, num_samples=100, num_labelled=10, gamma=0.5):
        # Number of samples
        self.num_samples = num_samples
        # Number of labelled samples
        self.num_labelled = num_labelled
        # Gamma parameter for RBF kernel
        self.gamma = gamma
        # Colors for two clusters
        self.color1 = 'red'
        self.color2 = 'green'
        # Color map
        self.cmap = ListedColormap([self.color1, self.color2])
        # Generate random 2D points for the two clusters
        self.cluster1 = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, 2))
        self.cluster2 = np.random.normal(loc=3.0, scale=1.0, size=(num_samples, 2))
        # Select labelled points from each cluster
        self.labelled_indices1, self.labelled_indices2, self.labelled_indices2_offset, self.unlabelled_indices = self.select_labelled_points()
        # Assign labels to the selected points
        self.labelled_points1 = self.cluster1[self.labelled_indices1]
        self.labelled_points2 = self.cluster2[self.labelled_indices2_offset]
        # Get unlabelled points
        self.unlabelled_points = np.vstack((self.cluster1, self.cluster2))[self.unlabelled_indices]
        # Assign true labels to the unlabelled points
        self.true_labels_unlabelled = np.concatenate([-1 * np.ones(len(self.cluster1) - len(self.labelled_indices1)),
                                                      np.ones(len(self.cluster2) - len(self.labelled_indices2))])
        # Compute similarity
        W, W_bar = self.compute_similarity(self.unlabelled_points, self.labelled_points1, self.labelled_points2)
        # Compute Hessian and gradient
        self.H, self.g = self.compute_Hessian_Gradient(W, W_bar)

    # Function to select labelled points from each cluster
    def select_labelled_points(self):
        # Select labelled indices for each cluster
        labelled_indices1 = random.sample(range(self.num_samples), self.num_labelled)
        labelled_indices2_offset = random.sample(range(self.num_samples), self.num_labelled)
        labelled_indices2 = [index + self.num_samples for index in labelled_indices2_offset]
        all_indices = list(range(2*self.num_samples))
        # Get unlabelled indices
        unlabelled_indices = [index for index in all_indices if index not in labelled_indices1 and index not in labelled_indices2]
        return labelled_indices1, labelled_indices2, labelled_indices2_offset, unlabelled_indices

    # Function to compute similarity between points using RBF kernel
    def compute_similarity(self, unlabelled_points, labelled_points1, labelled_points2):
        # Compute similarity between unlabelled points and labelled points for each cluster
        W_bar = rbf_kernel(unlabelled_points, unlabelled_points, gamma=self.gamma)
        W = rbf_kernel(unlabelled_points, np.vstack((labelled_points1, labelled_points2)), gamma=self.gamma)
        return W, W_bar

        # Function to compute Hessian and gradient

    def compute_Hessian_Gradient(self, W, W_bar):
        # Compute Hessian and linear term
        #TODO: define the Hessian H and the linear term g from the matrices W and W_bar
        H = ...
        g = ...
        return H, g

        # Function to plot clusters

    def plot_clusters(self, labels_pred):
        # Get labelled points and true labels for unlabelled points
        labelled_points1, labelled_points2, unlabelled_points = self.labelled_points1, self.labelled_points2, self.unlabelled_points
        true_labels_unlabelled = np.concatenate([-1 * np.ones(len(self.cluster1) - len(labelled_points1)),
                                                 np.ones(len(self.cluster2) - len(labelled_points2))])

        # Find indices of correct and incorrect labels
        correct_label_indices = np.where(labels_pred == true_labels_unlabelled)[0]
        incorrect_label_indices = np.where(labels_pred != true_labels_unlabelled)[0]

        # Set marker sizes, inversely proportional to number of points
        correct_marker_size = max(10, int(3000 / len(true_labels_unlabelled)))

        plt.figure(figsize=(10, 10))

        # 1) Plot original clusters
        plt.subplot(2, 2, 1)
        plt.scatter(self.cluster1[:, 0], self.cluster1[:, 1], color=self.color1, s=correct_marker_size)
        plt.scatter(self.cluster2[:, 0], self.cluster2[:, 1], color=self.color2, s=correct_marker_size)
        plt.title('Plot 1: Original Clusters')

        # 2) Highlight labeled points
        plt.subplot(2, 2, 2)
        plt.scatter(np.vstack((self.cluster1, self.cluster2))[:, 0], np.vstack((self.cluster1, self.cluster2))[:, 1],
                    color='black')
        plt.scatter(labelled_points1[:, 0], labelled_points1[:, 1], color=self.color1, s=correct_marker_size)
        plt.scatter(labelled_points2[:, 0], labelled_points2[:, 1], color=self.color2, s=correct_marker_size)
        plt.title('Plot 2: Labeled Points Highlighted')

        # 3) Show predicted labels
        plt.subplot(2, 2, 3)
        plt.scatter(unlabelled_points[:, 0], unlabelled_points[:, 1], c=labels_pred, cmap=self.cmap, alpha=0.5, s=correct_marker_size)
        plt.scatter(labelled_points1[:, 0], labelled_points1[:, 1], color=self.color1, s=correct_marker_size)
        plt.scatter(labelled_points2[:, 0], labelled_points2[:, 1], color=self.color2, s=correct_marker_size)
        plt.title('Plot 3: Predicted Labels')

        # 4) Plot showing correct and incorrect labels
        plt.subplot(2, 2, 4)
        plt.scatter(unlabelled_points[correct_label_indices, 0], unlabelled_points[correct_label_indices, 1],
                    c=labels_pred[correct_label_indices], cmap=self.cmap, marker='o', s=correct_marker_size, alpha=0.5)
        plt.scatter(unlabelled_points[incorrect_label_indices, 0], unlabelled_points[incorrect_label_indices, 1],
                    c=labels_pred[incorrect_label_indices], cmap=self.cmap, marker='x', s=correct_marker_size, alpha=0.5)
        plt.scatter(labelled_points1[:, 0], labelled_points1[:, 1], color=self.color1, s=correct_marker_size)
        plt.scatter(labelled_points2[:, 0], labelled_points2[:, 1], color=self.color2, s=correct_marker_size)
        plt.title('Plot 4: Correct and Incorrect Labels')

        # Adjust layout to avoid overlaps and display the plots
        plt.tight_layout()
        plt.show()