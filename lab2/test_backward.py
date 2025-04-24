
import numpy as np
import matplotlib.pyplot as plt
from lab2_proto import concatHMMs, backward
from lab2_tools import log_multivariate_normal_density_diag, logsumexp
from prondict import prondict

# Load phoneme HMMs and example
phoneHMMs = np.load("lab2_models_onespkr.npz", allow_pickle=True)["phoneHMMs"].item()
example = np.load("lab2_example.npz", allow_pickle=True)["example"].item()
X = example["lmfcc"]

# Build word-level HMM for digit 'o'
wordHMM = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
log_emlik = log_multivariate_normal_density_diag(X, wordHMM['means'], wordHMM['covars'])
log_startprob = np.log(wordHMM["startprob"])
log_transmat = np.log(wordHMM["transmat"])

# Compute backward probabilities
logbeta = backward(log_emlik, log_startprob, log_transmat)

# Validate against the example
match_logbeta = np.allclose(logbeta, example["logbeta"], atol=1e-4)
print("✅ logbeta match:", match_logbeta)

# Compute log-likelihood using beta (optional method)
loglik_beta = logsumexp(log_startprob[:-1] + log_emlik[0] + logbeta[0])
print("✅ loglik via beta matches example:", np.isclose(loglik_beta, example["loglik"], atol=1e-4))
print("   loglik via beta =", loglik_beta)

# Plot the logbeta heatmap
plt.figure(figsize=(12, 5))
plt.imshow(logbeta.T, aspect='auto', origin='lower', cmap='magma')
plt.colorbar(label="Log Backward Probability")
plt.title("Step 5.4 - Backward Probabilities (logbeta)")
plt.xlabel("Time Frame")
plt.ylabel("HMM State")
plt.tight_layout()
plt.show()