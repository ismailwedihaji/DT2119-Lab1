import numpy as np
import matplotlib.pyplot as plt
from lab2_proto import concatHMMs
from lab2_tools import log_multivariate_normal_density_diag
from prondict import prondict

# Step 1: Load phoneme models (trained on one speaker)
#phoneHMMs = np.load("lab2_models_onespkr.npz", allow_pickle=True)['phoneHMMs'].item()
phoneHMMs = np.load("lab2_models_all.npz", allow_pickle=True)['phoneHMMs'].item()

# Step 2: Build word-level HMM for digit 'o' = ['sil', 'ow', 'sil']
phoneme_sequence = ['sil', 'ow', 'sil']
wordHMM = concatHMMs(phoneHMMs, phoneme_sequence)

print("==== Word HMM for Digit 'o' ====")
print("Phoneme sequence:", phoneme_sequence)
print("Startprob shape:", wordHMM['startprob'].shape)
print("Transmat shape:", wordHMM['transmat'].shape)
print("Means shape:", wordHMM['means'].shape)
print("Covars shape:", wordHMM['covars'].shape)
print()

# Step 3: Load example utterance
example = np.load("lab2_example.npz", allow_pickle=True)['example'].item()
X = example['lmfcc']  # MFCC features (frames × 13)

# Step 4: Compute obsloglik (log likelihood of each frame under each state)
obsloglik = log_multivariate_normal_density_diag(X, wordHMM['means'], wordHMM['covars'])

# Step 5: Compare to precomputed reference
ref_obsloglik = example['obsloglik']
print("==== Comparison ====")
print("Shape match:", obsloglik.shape == ref_obsloglik.shape)
print("Values match (tolerance 1e-4):", np.allclose(obsloglik, ref_obsloglik, atol=1e-4))
print()

# Step 6: Optional - Visualize
plt.imshow(obsloglik.T, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
plt.colorbar(label="Log Likelihood")
plt.xlabel("Time Frame")
plt.ylabel("HMM State")
plt.title("Section 4.1 - Observation Log Likelihood (obsloglik)")
plt.tight_layout()
plt.show()
