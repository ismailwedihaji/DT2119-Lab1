import numpy as np
import os, sys
sys.path.append(os.path.abspath(".."))

from lab2_proto import forward, backward, statePosteriors, updateMeanAndVar, concatHMMs
from lab2_tools import log_multivariate_normal_density_diag
from prondict import prondict

# Load data and models
data = np.load('../lab2_data.npz', allow_pickle=True)['data']
phoneHMMs = np.load('../lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
X = data[10]['lmfcc']
wordHMM = concatHMMs(phoneHMMs, ['sil'] + prondict['4'] + ['sil'])

# Parameters
threshold = 1.0
max_iter = 20
prev_loglik = -np.inf

for it in range(max_iter):
    log_emlik = log_multivariate_normal_density_diag(X, wordHMM['means'], wordHMM['covars'])
    log_startprob = np.log(wordHMM['startprob'])
    log_transmat = np.log(wordHMM['transmat'])

    logalpha = forward(log_emlik, log_startprob, log_transmat)
    logbeta = backward(log_emlik, log_startprob, log_transmat)
    loggamma = statePosteriors(logalpha, logbeta)

    # Compute new mean and var
    wordHMM['means'], wordHMM['covars'] = updateMeanAndVar(X, loggamma)

    # Compute new log-likelihood
    new_loglik = np.logaddexp.reduce(logalpha[-1])

    print(f"Iteration {it+1}, Log-likelihood: {new_loglik:.2f}")

    if new_loglik - prev_loglik < threshold:
        break
    prev_loglik = new_loglik
