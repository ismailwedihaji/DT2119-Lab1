import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lab2")))
from lab3_tools import *
from lab2_proto import concatHMMs, viterbi
from lab2_tools import log_multivariate_normal_density_diag

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """
    Converts word-level transcription into phone-level transcription,
    adding optional silence and short pauses.

    Args:
       wordList: list of word strings (e.g. ['four', 'three'])
       pronDict: dictionary mapping words to phoneme lists
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model 'sp' between words

    Returns:
       List of phonemes, e.g. ['sil', 'f', 'ao', 'r', 'sp', 'th', 'r', 'iy', 'sil']
    """
    phoneList = []

    for i, word in enumerate(wordList):
        word = word.lower()
        if word not in pronDict:
            raise ValueError(f"Word '{word}' not in pronunciation dictionary.")

        phones = pronDict[word]
        phoneList.extend(phones)

        if addShortPause and i != len(wordList) - 1:
            phoneList.append('sp')

    if addSilence:
        phoneList = ['sil'] + phoneList + ['sil']

    return phoneList


def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """
    Performs forced alignment of MFCC features to a phoneme-level HMM sequence.

    Args:
       lmfcc: NxD array of MFCC feature vectors
       phoneHMMs: dictionary of phoneme HMMs
       phoneTrans: list of phoneme strings (e.g., ['sil', 'z', 'iy', ..., 'sil'])

    Returns:
       List of aligned state names (e.g., ['z_0', 'z_1', 'z_2', ...])
    """
    # 1. Build utterance HMM by concatenating individual phone HMMs
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

    # 2. Compute observation log-likelihoods
    obsloglik = log_multivariate_normal_density_diag(
        lmfcc,
        utteranceHMM['means'],
        utteranceHMM['covars']
    )

    # 3. Run Viterbi algorithm to find best state path
    _, state_seq = viterbi(
        obsloglik,
        utteranceHMM['startprob'],
        utteranceHMM['transmat']
    )

    # 4. Construct full list of state labels
    state_labels = []
    for ph in phoneTrans:
        n_states = phoneHMMs[ph]['means'].shape[0]
        for i in range(n_states):
            state_labels.append(f"{ph}_{i}")

    # 5. Map Viterbi state indices to their state labels
    aligned = [state_labels[s] for s in state_seq]

    return aligned