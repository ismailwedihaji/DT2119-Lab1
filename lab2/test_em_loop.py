
import numpy as np
import matplotlib.pyplot as plt
from lab2_proto import concatHMMs, forward, backward, statePosteriors, updateMeanAndVar
from lab2_tools import log_multivariate_normal_density_diag, logsumexp
from prondict import prondict

# Load phoneme HMMs and example
phoneHMMs = np.load("lab2_models_onespkr.npz", allow_pickle=True)["phoneHMMs"].item()
example = np.load("lab2_example.npz", allow_pickle=True)["example"].item()
X = example["lmfcc"]

# Build word-level HMM for digit 'o'
wordHMM = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
means, covars = wordHMM['means'], wordHMM['covars']
log_startprob = np.log(wordHMM["startprob"])
log_transmat = np.log(wordHMM["transmat"])

# EM loop parameters
max_iters = 20
threshold = 1.0
logliks = []
prev_loglik = -np.inf

# EM loop
for i in range(max_iters):
    log_emlik = log_multivariate_normal_density_diag(X, means, covars)
    logalpha = forward(log_emlik, log_startprob, log_transmat)
    logbeta = backward(log_emlik, log_startprob, log_transmat)
    loggamma = statePosteriors(logalpha, logbeta)

    # M-step
    means, covars = updateMeanAndVar(X, loggamma)

    # E-step: log-likelihood
    loglik = logsumexp(logalpha[-1])
    logliks.append(loglik)
    print(f"Iteration {i+1:2d}, Log-Likelihood: {loglik:.4f}")

    # Convergence check
    if loglik - prev_loglik < threshold:
        print("✅ Converged")
        break
    prev_loglik = loglik

# Plot log-likelihood progression
plt.figure(figsize=(10, 5))
plt.plot(logliks, marker='o')
plt.title("Log-Likelihood over EM Iterations")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.grid(True)
plt.tight_layout()
plt.show()