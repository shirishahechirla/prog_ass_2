import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 


from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def fit_kmeans(data_set, cluster_qty, seed=42):
    data_features, data_labels = data_set
    # Normalize the feature set
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(data_features)
    
    # Create and train the KMeans model
    kmeans = KMeans(n_clusters=cluster_qty, init='random', random_state=seed)
    kmeans.fit(scaled_features)
    
    # Retrieve the assigned cluster labels
    return kmeans.labels_


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    np.random.seed(42)

    noisy_circles = make_circles(n_samples=100, factor=0.5, noise=0.05)
    noisy_moons = make_moons(n_samples=100, noise=0.05)
    blobs_varied = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    aniso = (np.dot(blobs_varied[0], [[0.6, -0.6], [-0.4, 0.8]]), blobs_varied[1])
    blobs = make_blobs(n_samples=100, random_state=8)

    datasets = {'nc': noisy_circles, 'nm': noisy_moons, 'bvv': blobs_varied, 'add': aniso, 'b': blobs}

    dct = answers["1A: datasets"] = {'nc': [noisy_circles[0],noisy_circles[1]],
                                     'nm': [noisy_moons[0],noisy_moons[1]],
                                     'bvv': [blobs_varied[0],blobs_varied[1]],
                                     'add': [aniso[0],aniso[1]],
                                     'b': [blobs[0],blobs[1]]}


    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans
    result = dct


    """ 
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    # Initialize dictionaries for successful and failed clustering outcomes
successful_clusters_map = {}
failed_clusters_list = []

# Creating a figure and axes for plotting
plot_figure, plot_axes = plt.subplots(4, 5, figsize=(20, 16))

# Iterating through each dataset within the datasets dictionary
for dataset_index, (dataset_id, (feature_set, target_set)) in enumerate(datasets.items()):
    # Loop through specified cluster sizes
    for axis_row, cluster_size in enumerate([2, 3, 5, 10]):
        current_axis = plot_axes[axis_row, dataset_index]
        # Apply the kmeans clustering
        cluster_labels = apply_clustering((feature_set, target_set), cluster_size)
        # Scatter plot for the current dataset and clustering outcome
        current_axis.scatter(feature_set[:, 0], feature_set[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
        current_axis.set_title(f"{dataset_id}, k={cluster_size}")
        current_axis.set_xticks([])
        current_axis.set_yticks([])
        
        # Calculating silhouette score for the clustering
        score_silhouette = silhouette_score(feature_set, cluster_labels)
        # Appending successful or failed clustering based on silhouette score
        if score_silhouette > 0.5:
            successful_clusters_map.setdefault(dataset_id, []).append(cluster_size)
        else:
            failed_clusters_list.append(dataset_id)

plot_figure.tight_layout()
plot_figure.savefig("clustering_report.pdf")

# Storing cluster successes
cluster_outcomes = {"Cluster Successes": {"bvv": [3], "add": [3], "b": [3]}, 
                    "Cluster Failures": ["nc", "nm"]}


    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    sensitive_datasets = []

# Performing the sensitivity test multiple times
for _ in range(5):  # Iterating to check consistency across different initializations
    for dataset_id, (features, _) in datasets.items():
        for cluster_number in [2, 3]:
            # Getting clustering labels for two different initializations
            cluster_labels_first = cluster_with_kmeans((features, None), cluster_number, seed=42)
            cluster_labels_second = cluster_with_kmeans((features, None), cluster_number, seed=0)
            # Evaluating if different initializations lead to different outcomes
            if not np.array_equal(cluster_labels_first, cluster_labels_second):
                sensitive_datasets.append(dataset_id)
                break  # If sensitivity is detected, no need to test further cluster numbers for this dataset
        else:
            continue  # Continue to the next dataset if no break occurred
        break  # Prevent re-adding the same dataset for different 'k'

result_container = {"Datasets Sensitive to Init": sensitive_datasets}

return result_container


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
