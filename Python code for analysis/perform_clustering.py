"""
===========================================
Train topic model
===========================================

"""

# Author: Hao Zhang
# License: Apache License Version 2.0


import numpy as np
import scipy.io as sio
import umap
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

data = sio.loadmat('../trained_topic_model/PFA_trained_model.mat')
Topic_proportion = data['Theta_mean'].T
Topic_proportion = Topic_proportion/Topic_proportion.sum(1)[:,np.newaxis]

from sklearn import preprocessing  # to normalise existing X
Topic_proportion = preprocessing.normalize(Topic_proportion)

print('UMAP')
UMAP = umap.UMAP(metric='euclidean', n_neighbors=30, random_state=0)
embedding = UMAP.fit_transform(Topic_proportion)


print('Hierarhical clustering')
num_cluster = 4
cluster = AgglomerativeClustering(n_clusters=num_cluster, affinity='euclidean', linkage='ward')
cluster.fit_predict(Topic_proportion)

plt.figure()
plt.scatter(embedding[cluster.labels_==0, 0], embedding[cluster.labels_==0, 1], c='#fe7f0e', s=5)
plt.scatter(embedding[cluster.labels_==1, 0], embedding[cluster.labels_==1, 1], c='#9467bc', s=5)
plt.scatter(embedding[cluster.labels_==2, 0], embedding[cluster.labels_==2, 1], c='#2ba02d', s=5)
plt.scatter(embedding[cluster.labels_==3, 0], embedding[cluster.labels_==3, 1], c='#d52727', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.savefig('./result/UMAP.jpg', dpi=600, format='jpg')

plt.title("Hierarchical Clustering Dendrogram")
plt.figure()
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model.fit(Topic_proportion)
plot_dendrogram(model)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.savefig('./result/Dendrogram.jpg', dpi=600, format='jpg')