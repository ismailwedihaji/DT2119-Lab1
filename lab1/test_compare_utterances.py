import numpy as np
import matplotlib.pyplot as plt
from lab1_proto import mfcc, dtw
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, dendrogram
from lab1_tools import tidigit2labels

# Load data
data = np.load("lab1_data.npz", allow_pickle=True)["data"]

# Step 1: Extract MFCCs for each utterance
mfcc_features = [mfcc(utt["samples"]) for utt in data]

# Step 2: Compute DTW distance matrix (44x44)
D = np.zeros((44, 44))
for i in range(44):
    for j in range(i, 44):  # matrix is symmetric
        d, _, _ = dtw(mfcc_features[i], mfcc_features[j], euclidean)
        D[i, j] = d
        D[j, i] = d

# Step 3: Show distance matrix with pcolormesh
plt.figure(figsize=(6, 5))
plt.pcolormesh(D, cmap="viridis", shading="auto")
plt.title("DTW Distance Matrix (44x44)")
plt.xlabel("Utterance Index")
plt.ylabel("Utterance Index")
plt.colorbar()
plt.tight_layout()
plt.show()

# Step 4: Hierarchical clustering
Z = linkage(D, method="complete")
labels = tidigit2labels(data)

plt.figure(figsize=(10, 5))
dendrogram(Z, labels=labels, leaf_rotation=90)
plt.title("Hierarchical Clustering of Digits (DTW distances)")
plt.tight_layout()
plt.show()
