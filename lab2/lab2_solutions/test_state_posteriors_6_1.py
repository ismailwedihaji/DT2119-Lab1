import numpy as np
import os, sys
sys.path.append(os.path.abspath(".."))

from lab2_proto import forward, backward, statePosteriors, concatHMMs
from lab2_tools import log_multivariate_normal_density_diag

example = np.load('../lab2_example.npz', allow_pickle=True)['example'].item()
phoneHMMs = np.load('../lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
wordHMM = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])


X = example['lmfcc']
log_emlik = log_multivariate_normal_density_diag(X, wordHMM['means'], wordHMM['covars'])
log_startprob = np.log(wordHMM['startprob'])
log_transmat = np.log(wordHMM['transmat'])

# Compute forward and backward probabilities
logalpha = forward(log_emlik, log_startprob, log_transmat)
logbeta = backward(log_emlik, log_startprob, log_transmat)

# Compute state posterior probabilities (log-gamma)
loggamma = statePosteriors(logalpha, logbeta)

# Compare with reference gamma from example
print("Shape match:", loggamma.shape == example['loggamma'].shape)
print("Content match:", np.allclose(loggamma, example['loggamma'], atol=1e-4))

# Verify posteriors sum to 1 across states (after exponentiating)
print("Posterior sums (should all be ≈1):")
print(np.sum(np.exp(loggamma), axis=1))
