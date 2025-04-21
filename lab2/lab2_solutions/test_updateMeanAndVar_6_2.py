import numpy as np
import os, sys
sys.path.append(os.path.abspath(".."))

from lab2_proto import forward, backward, statePosteriors, updateMeanAndVar, concatHMMs
from lab2_tools import log_multivariate_normal_density_diag

# Load example utterance and model
example = np.load('../lab2_example.npz', allow_pickle=True)['example'].item()
phoneHMMs = np.load('../lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
wordHMM = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])

# Input features and model parameters
X = example['lmfcc']
log_emlik = log_multivariate_normal_density_diag(X, wordHMM['means'], wordHMM['covars'])
log_startprob = np.log(wordHMM['startprob'])
log_transmat = np.log(wordHMM['transmat'])

# Compute forward, backward, and gamma
logalpha = forward(log_emlik, log_startprob, log_transmat)
logbeta = backward(log_emlik, log_startprob, log_transmat)
loggamma = statePosteriors(logalpha, logbeta)

# Run updateMeanAndVar
means, covars = updateMeanAndVar(X, loggamma, varianceFloor=5.0)

# Print shape of output to verify
print("Means shape:", means.shape)
print("Covars shape:", covars.shape)

# Check that no variances are below the variance floor
print("Minimum variance value:", np.min(covars))
if np.any(covars < 5.0):
    print("Warning: Some variances are below the floor.")
else:
    print("All variances respect the variance floor.")
