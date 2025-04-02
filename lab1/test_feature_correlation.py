import numpy as np
import matplotlib.pyplot as plt
from lab1_proto import mfcc, mspec

# Step 1: Load the data
data = np.load("lab1_data.npz", allow_pickle=True)["data"]

# Step 2: Compute MFCCs and Mel Spectrums for all utterances
mfcc_features = [mfcc(utt["samples"]) for utt in data]
mspec_features = [mspec(utt["samples"]) for utt in data]

# Step 3: Stack all frames into one big matrix
all_mfcc_frames = np.vstack(mfcc_features)     # Shape: [total_frames x 13]
all_mspec_frames = np.vstack(mspec_features)   # Shape: [total_frames x num_filters]

# Step 4: Compute correlation matrices
mfcc_corr = np.corrcoef(all_mfcc_frames.T)     # [13 x 13]
mspec_corr = np.corrcoef(all_mspec_frames.T)   # [num_filters x num_filters]

# Step 5: Plot correlation matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.pcolormesh(mfcc_corr, cmap='coolwarm', shading='auto')
plt.title("MFCC Feature Correlation")
plt.xlabel("Coefficient Index")
plt.ylabel("Coefficient Index")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.pcolormesh(mspec_corr, cmap='coolwarm', shading='auto')
plt.title("Mel Filterbank Feature Correlation")
plt.xlabel("Filter Index")
plt.ylabel("Filter Index")
plt.colorbar()

plt.tight_layout()
plt.show()
