import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(".."))

from lab2_tools import log_multivariate_normal_density_diag
from lab2_proto import concatHMMs


# Step 1: Load phone-level HMMs and the example utterance
phoneHMMs = np.load('../lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
example = np.load('../lab2_example.npz', allow_pickle=True)['example'].item()

# Step 2: Build word-level HMM for digit 'o' using pronunciation ['sil', 'ow', 'sil']
wordHMM = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])

# Step 3: Extract input features (MFCCs) from the example
X = example['lmfcc']  

# Step 4: Extract HMM parameters
means = wordHMM['means']     
covars = wordHMM['covars']   

# Step 5: Compute log-likelihood of each frame for each state
obsloglik = log_multivariate_normal_density_diag(X, means, covars)


#step 6: plot the result
plt.imshow(obsloglik.T, aspect='auto', origin='lower', interpolation='none')
plt.colorbar(label='Log Likelihood')
plt.xlabel('Time Frame')
plt.ylabel('HMM State')
plt.title("obsloglik: HMM log likelihood of observation given the state")
plt.tight_layout()
plt.show()

# Step 7: Compare with precomputed values
print("Shape of obsloglik:", obsloglik.shape)
print("Shape of example obsloglik:", example['obsloglik'].shape)
print("Match with example:", np.allclose(obsloglik, example['obsloglik'], atol=1e-4))
print(obsloglik)
