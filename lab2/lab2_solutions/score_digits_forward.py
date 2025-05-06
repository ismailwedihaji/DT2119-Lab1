import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.append(os.path.abspath(".."))  # Fix import path for parent folder

from lab2_tools import logsumexp, log_multivariate_normal_density_diag
from lab2_proto import concatHMMs, forward
from prondict import prondict  

# Load phone-level HMMs and all utterance data
phoneHMMs = np.load('../lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
#phoneHMMs = np.load('../lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()

data = np.load('../lab2_data.npz', allow_pickle=True)['data']

# Build word-level HMMs for each digit (0–9) using phoneme sequences from prondict
digit_list = list(prondict.keys())  # This gives you ['o', 'z', '1', ..., '9']
wordHMMs = {
    digit: concatHMMs(phoneHMMs, ['sil'] + prondict[digit] + ['sil'])
    for digit in digit_list
}

# Score each of the 44 utterances using all 11 HMMs and choose the most likely digit
correct = 0
for i, utterance in enumerate(data):
    X = utterance['lmfcc']
    true_digit = utterance['digit']
    
    best_loglik = -np.inf
    best_digit = None

    for digit in digit_list:
        hmm = wordHMMs[digit]
        log_emlik = log_multivariate_normal_density_diag(X, hmm['means'], hmm['covars'])
        logalpha = forward(log_emlik, np.log(hmm['startprob']), np.log(hmm['transmat']))
        loglik = logsumexp(logalpha[-1])  # Total sequence log-likelihood

        if loglik > best_loglik:
            best_loglik = loglik
            best_digit = digit

    if best_digit == true_digit:
        correct += 1

    print(f"Utterance {i+1}: True = {true_digit}, Predicted = {best_digit}")

# Print overall recognition accuracy
accuracy = correct / len(data)
print(f"\nTotal Accuracy: {accuracy * 100:.2f}%")
