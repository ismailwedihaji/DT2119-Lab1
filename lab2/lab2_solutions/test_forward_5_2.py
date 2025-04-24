import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))

from lab2_proto import forward, concatHMMs
from lab2_tools import logsumexp

phoneHMMs = np.load('../lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
example = np.load('../lab2_example.npz', allow_pickle=True)['example'].item()

# Step 1: Build word-level HMM for digit 'o'
wordHMM = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])

# Step 2: Prepare log parameters and input
log_startprob = np.log(wordHMM['startprob'])
log_transmat = np.log(wordHMM['transmat'])
log_emlik = example['obsloglik']  # from Section 5.1

# Step 3: Run your forward implementation
logalpha = forward(log_emlik, log_startprob, log_transmat)

plt.imshow(logalpha.T, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Log Alpha (Forward Probabilities)')
plt.title('Forward Algorithm: Log Alpha')
plt.xlabel('Time Frame')
plt.ylabel('HMM State')
plt.show()

# Step 4: Compare with example
print("Shape match:", logalpha.shape == example['logalpha'].shape)
print("Content match:", np.allclose(logalpha, example['logalpha'], atol=1e-4))

# Step 5: Compute total log-likelihood from logalpha
loglik = logsumexp(logalpha[-1])
print("Log-likelihood match:", np.isclose(loglik, example['loglik'], atol=1e-4))


