
import numpy as np
import matplotlib.pyplot as plt
from lab2_proto import concatHMMs, forward, backward, statePosteriors, updateMeanAndVar
from lab2_tools import log_multivariate_normal_density_diag
from prondict import prondict

# Load phoneme HMMs and example
phoneHMMs = np.load("lab2_models_onespkr.npz", allow_pickle=True)["phoneHMMs"].item()
example = np.load("lab2_example.npz", allow_pickle=True)["example"].item()
X = example["lmfcc"]

# Build word-level HMM for digit 'o'
wordHMM = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
log_emlik = log_multivariate_normal_density_diag(X, wordHMM["means"], wordHMM["covars"])
log_startprob = np.log(wordHMM["startprob"])
log_transmat = np.log(wordHMM["transmat"])

# Compute state posteriors (gamma)
logalpha = forward(log_emlik, log_startprob, log_transmat)
logbeta = backward(log_emlik, log_startprob, log_transmat)
loggamma = statePosteriors(logalpha, logbeta)

# Retrain means and covars
new_means, new_covars = updateMeanAndVar(X, loggamma)

# Print changes for inspection
print("✅ Means and covariances updated successfully")
print("Old means shape:", wordHMM['means'].shape)
print("New means shape:", new_means.shape)
print("Old covars shape:", wordHMM['covars'].shape)
print("New covars shape:", new_covars.shape)

# Optional: visualize mean change (first dimension only)
plt.figure(figsize=(12, 5))
plt.plot(wordHMM['means'][:, 0], label="Old Means (dim 0)", linestyle='--')
plt.plot(new_means[:, 0], label="New Means (dim 0)", linestyle='-')
plt.title("Comparison of First-Dimension Means Before and After EM Update")
plt.xlabel("HMM State")
plt.ylabel("Mean Value (dimension 0)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()