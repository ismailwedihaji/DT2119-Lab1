import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from lab1_proto import mfcc  # Your own MFCC function

# Step 1: Load the dataset
data = np.load("lab1_data.npz", allow_pickle=True)["data"]

# Step 2: Extract MFCCs from all utterances
mfcc_all = [mfcc(utt["samples"]) for utt in data]  # Each element is [frames x 13]

# Step 3: Concatenate all MFCC frames into a single matrix
X = np.vstack(mfcc_all)  # Shape: [total_frames x 13]

# Step 4: Train a GMM with 32 components on all MFCC frames
gmm = GaussianMixture(n_components=32, covariance_type='diag', random_state=42)
gmm.fit(X)

# Step 5: Analyze utterances containing the word "seven" (IDs 16, 17, 38, 39)
 seven_ids= [16, 17, 38, 39]

plt.figure(figsize=(12, 8))

for i, utt_id in enumerate(seven_ids):
    mfcc_feat = mfcc(data[utt_id]["samples"])               # Extract MFCC for the utterance
    posteriors = gmm.predict_proba(mfcc_feat)               # [frames x 32] posterior matrix

    plt.subplot(2, 2, i + 1)
    plt.pcolormesh(posteriors.T, cmap='viridis', shading='auto')  # Plot [32 x frames]
    plt.title(f"GMM Posterior for Utterance {utt_id}")
    plt.xlabel("Frame Index")
    plt.ylabel("Cluster Index")
    plt.colorbar()

plt.tight_layout()
plt.show()
