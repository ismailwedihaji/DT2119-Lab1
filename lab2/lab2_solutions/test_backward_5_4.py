import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))
from lab2_proto import backward, concatHMMs
from lab2_tools import logsumexp

# Load data and model
phoneHMMs = np.load('../lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
example = np.load('../lab2_example.npz', allow_pickle=True)['example'].item()

# Build HMM for digit "o"
wordHMM = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
log_startprob = np.log(wordHMM['startprob'])
log_transmat = np.log(wordHMM['transmat'])
log_emlik = example['obsloglik']  # reuse from 5.1

# Run backward algorithm
logbeta = backward(log_emlik, log_startprob, log_transmat)

# Visualize result
plt.imshow(logbeta.T, aspect='auto', origin='lower', cmap='plasma')
plt.colorbar(label='Log Beta (Backward Probabilities)')
plt.title('Backward Algorithm: Log Beta')
plt.xlabel('Time Frame')
plt.ylabel('HMM State')
plt.show()

# Check shape and values
print("Shape match:", logbeta.shape == example['logbeta'].shape)
print("Content match:", np.allclose(logbeta, example['logbeta'], atol=1e-4))

# Compute log-likelihood using beta method (optional)
loglik = logsumexp(log_startprob[:log_emlik.shape[1]] + log_emlik[0] + logbeta[0])
print("Log-likelihood match:", np.isclose(loglik, example['loglik'], atol=1e-4))
