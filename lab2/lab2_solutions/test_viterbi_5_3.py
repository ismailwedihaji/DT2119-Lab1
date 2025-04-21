import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(".."))

from lab2_tools import log_multivariate_normal_density_diag
from lab2_proto import viterbi, concatHMMs

# Load data
example = np.load('../lab2_example.npz', allow_pickle=True)['example'].item()
phoneHMMs = np.load('../lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
wordHMM = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])

X = example['lmfcc']
log_emlik = log_multivariate_normal_density_diag(X, wordHMM['means'], wordHMM['covars'])

vloglik, path = viterbi(
    log_emlik,
    np.log(wordHMM['startprob']),
    np.log(wordHMM['transmat'])
)

print("Log-likelihood match:", np.isclose(vloglik, example['vloglik'], atol=1e-4))

# Optional: visualize path on top of forward probabilities
plt.imshow(example['logalpha'].T, origin='lower', aspect='auto', cmap='viridis')
plt.plot(path, color='red', linewidth=1.0)
plt.title("Viterbi path on top of logalpha")
plt.xlabel("Time Frame")
plt.ylabel("State")
plt.colorbar(label="Logalpha")
plt.show()
