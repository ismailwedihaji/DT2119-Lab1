import numpy as np
import matplotlib.pyplot as plt
from lab2_proto import concatHMMs, forward, viterbi
from lab2_tools import log_multivariate_normal_density_diag, logsumexp
from prondict import prondict

# Load phoneme HMMs and example
phoneHMMs = np.load("lab2_models_onespkr.npz", allow_pickle=True)["phoneHMMs"].item()
example = np.load("lab2_example.npz", allow_pickle=True)["example"].item()
X = example["lmfcc"]  # MFCCs of the utterance

# Build word-level HMM for digit 'o' = ['sil', 'ow', 'sil']
wordHMM = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
means, covars = wordHMM["means"], wordHMM["covars"]
log_startprob = np.log(wordHMM["startprob"])
log_transmat = np.log(wordHMM["transmat"])

# === Step 5.1: Observation log-likelihoods ===
log_emlik = log_multivariate_normal_density_diag(X, means, covars)
print("✅ obsloglik match:", np.allclose(log_emlik, example["obsloglik"], atol=1e-4))

plt.figure(figsize=(12, 5))
plt.imshow(log_emlik.T, aspect='auto', origin='lower', cmap='plasma')
plt.colorbar(label="Log Likelihood")
plt.title("Step 5.1 - Observation Log-Likelihood (obsloglik)")
plt.xlabel("Time Frame")
plt.ylabel("HMM State")
plt.tight_layout()
plt.show()

# === Step 5.2: Forward Algorithm ===
logalpha = forward(log_emlik, log_startprob, log_transmat)
loglik = logsumexp(logalpha[-1])
print("✅ logalpha match:", np.allclose(logalpha, example["logalpha"], atol=1e-4))
print("✅ loglik match:", np.isclose(loglik, example["loglik"], atol=1e-4))
print("   loglik =", loglik)

plt.figure(figsize=(12, 5))
plt.imshow(logalpha.T, aspect='auto', origin='lower', cmap='inferno')
plt.colorbar(label="Log Forward Probability")
plt.title("Step 5.2 - Forward Probabilities (logalpha)")
plt.xlabel("Time Frame")
plt.ylabel("HMM State")
plt.tight_layout()
plt.show()

# === Step 5.3: Viterbi Decoding ===
vloglik, viterbi_path = viterbi(log_emlik, log_startprob, log_transmat)
print("✅ vloglik match:", np.isclose(vloglik, example["vloglik"], atol=1e-4))
print("   vloglik =", vloglik)

plt.figure(figsize=(12, 5))
plt.imshow(logalpha.T, aspect='auto', origin='lower', cmap='inferno')
plt.plot(viterbi_path, color='cyan', linewidth=2, label='Viterbi Path')
plt.colorbar(label="Log Forward Probability")
plt.title("Step 5.3 - Viterbi Path Over Forward Probabilities")
plt.xlabel("Time Frame")
plt.ylabel("HMM State")
plt.legend()
plt.tight_layout()
plt.show()
