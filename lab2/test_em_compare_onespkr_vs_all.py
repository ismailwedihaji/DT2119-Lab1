
import numpy as np
import matplotlib.pyplot as plt
from lab2_proto import concatHMMs, forward, backward, statePosteriors, updateMeanAndVar
from lab2_tools import log_multivariate_normal_density_diag, logsumexp

def run_em_loop(phoneHMMs, phoneme_seq, X, label):
    wordHMM = concatHMMs(phoneHMMs, phoneme_seq)
    means, covars = wordHMM['means'], wordHMM['covars']
    log_startprob = np.log(wordHMM["startprob"])
    log_transmat = np.log(wordHMM["transmat"])

    max_iters = 20
    threshold = 1.0
    logliks = []
    prev_loglik = -np.inf

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
        print(f"{label} - Iteration {i+1:2d}, Log-Likelihood: {loglik:.4f}")

        if loglik - prev_loglik < threshold:
            print(f"{label} - ✅ Converged")
            break
        prev_loglik = loglik

    return logliks

# Load the example utterance
example = np.load("lab2_example.npz", allow_pickle=True)["example"].item()
X = example["lmfcc"]
phoneme_seq = ['sil', 'ow', 'sil']

# Run EM loop with two different initial models
onespkr_HMMs = np.load("lab2_models_onespkr.npz", allow_pickle=True)["phoneHMMs"].item()
all_HMMs = np.load("lab2_models_all.npz", allow_pickle=True)["phoneHMMs"].item()

logliks_one = run_em_loop(onespkr_HMMs, phoneme_seq, X, label="One Speaker")
logliks_all = run_em_loop(all_HMMs, phoneme_seq, X, label="All Speakers")

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(logliks_one, label="One Speaker Init", marker='o')
plt.plot(logliks_all, label="All Speakers Init", marker='s')
plt.title("EM Log-Likelihood Comparison: One Speaker vs All Speakers Init")
plt.xlabel("EM Iteration")
plt.ylabel("Log-Likelihood")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()