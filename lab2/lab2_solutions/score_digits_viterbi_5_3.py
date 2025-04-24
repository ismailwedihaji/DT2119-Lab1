import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.append(os.path.abspath(".."))

from lab2_tools import log_multivariate_normal_density_diag
from lab2_proto import concatHMMs, viterbi
from prondict import prondict  


phoneHMMs = np.load('../lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
#phoneHMMs = np.load('../lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
data = np.load('../lab2_data.npz', allow_pickle=True)['data']


digit_list = list(prondict.keys())  
wordHMMs = {
    digit: concatHMMs(phoneHMMs, ['sil'] + prondict[digit] + ['sil'])
    for digit in digit_list
}

# Viterbi digit recognition for all utterances
correct = 0
for i, utterance in enumerate(data):
    X = utterance['lmfcc']
    true_digit = utterance['digit']

    best_loglik = -np.inf
    best_digit = None

    for digit in digit_list:
        hmm = wordHMMs[digit]
        log_emlik = log_multivariate_normal_density_diag(X, hmm['means'], hmm['covars'])
        viterbi_loglik, _ = viterbi(
            log_emlik, np.log(hmm['startprob']), np.log(hmm['transmat'])
        )

        if viterbi_loglik > best_loglik:
            best_loglik = viterbi_loglik
            best_digit = digit

    if best_digit == true_digit:
        correct += 1

    print(f"Utterance {i+1}: True = {true_digit}, Predicted = {best_digit}")

# Final recognition accuracy using Viterbi
accuracy = correct / len(data)
print(f"\nTotal Viterbi Accuracy: {accuracy * 100:.2f}%")
