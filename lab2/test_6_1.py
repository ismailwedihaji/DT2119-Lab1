
import numpy as np
import matplotlib.pyplot as plt
from lab2_proto import concatHMMs, forward, backward, statePosteriors
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

# Compute forward and backward probabilities
logalpha = forward(log_emlik, log_startprob, log_transmat)
logbeta = backward(log_emlik, log_startprob, log_transmat)

# Compute state posterior probabilities (gamma)
loggamma = statePosteriors(logalpha, logbeta)

# Convert to linear domain for sum checks
gamma = np.exp(loggamma)
sum_over_states = np.sum(gamma, axis=1)
sum_over_all = np.sum(gamma)

print("✅ Each time step sums to ~1:", np.allclose(sum_over_states, 1, atol=1e-3))
print("🔢 Sum over all gamma values:", sum_over_all)
print("📏 Number of time frames:", gamma.shape[0])
print("\n🔍 Posterior sums per frame (should be ~1):")
print("-" * 40)
print(" Frame | Posterior Sum")
print("-" * 40)
for i, s in enumerate(sum_over_states):
    print(f" {i:5d} | {s:.6f}")
print("-" * 40)

# Plot gamma matrix
plt.figure(figsize=(12, 5))
plt.imshow(gamma.T, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label="State Posterior Probability")
plt.title("Step 6.1 - State Posterior Probabilities (gamma)")
plt.xlabel("Time Frame")
plt.ylabel("HMM State")
plt.tight_layout()
plt.show()