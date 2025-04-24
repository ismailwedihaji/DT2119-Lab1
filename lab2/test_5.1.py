import numpy as np
import matplotlib.pyplot as plt
from lab2_proto import concatHMMs
from lab2_tools import log_multivariate_normal_density_diag
from prondict import prondict

# Load phoneme HMMs and example
phoneHMMs = np.load("lab2_models_onespkr.npz", allow_pickle=True)['phoneHMMs'].item()
example = np.load("lab2_example.npz", allow_pickle=True)['example'].item()

# Step 1: Build word-level HMM for digit 'o'
wordHMM = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])

# Step 2: Get MFCCs (X) from example file
X = example['lmfcc']

# Step 3: Get Gaussian parameters
means = wordHMM['means']      # shape: (num_states, 13)
covars = wordHMM['covars']    # shape: (num_states, 13)

# Step 4: Compute observation log-likelihoods
obsloglik = log_multivariate_normal_density_diag(X, means, covars)

# Step 5: Compare with the example file
match = np.allclose(obsloglik, example['obsloglik'], atol=1e-4)
print("✅ Match with example['obsloglik']:", match)

# Step 6: Visualize
plt.imshow(obsloglik.T, aspect='auto', origin='lower', interpolation='none', cmap='plasma')
plt.colorbar(label="Log Likelihood")
plt.title("Step 5.1 - Log-Likelihood of Frames per HMM State")
plt.xlabel("Time Frame")
plt.ylabel("HMM State")
plt.tight_layout()
plt.show()

# Set up HMM structure for 'o' → ['sil', 'ow', 'sil']
phonemes = ['sil', 'ow', 'sil']
states_per_phoneme = 3
state_labels = []
for p in phonemes:
    state_labels.extend([f'{p}_0', f'{p}_1', f'{p}_2'])

# Plot with improved labeling
plt.figure(figsize=(12, 6))
plt.imshow(obsloglik.T, aspect='auto', origin='lower', interpolation='none', cmap='plasma')
plt.colorbar(label="Log Likelihood")
plt.xlabel("Time Frame")
plt.ylabel("HMM State")

# Add horizontal grid lines between phonemes
for i in range(0, len(state_labels) + 1, states_per_phoneme):
    plt.axhline(i - 0.5, color='white', linestyle='--', linewidth=0.5)

# Y-axis with state labels
plt.yticks(ticks=np.arange(len(state_labels)), labels=state_labels)

# Optional: vertical grid every 10 frames
for x in range(0, obsloglik.shape[0], 10):
    plt.axvline(x, color='white', linestyle=':', linewidth=0.2)

plt.title("Enhanced HMM State Activity Heatmap for Digit 'o'")
plt.tight_layout()
plt.show()
