import numpy as np
from prondict import prondict
from lab2_proto import concatHMMs
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load data and models
data = np.load('lab2_data.npz', allow_pickle=True)['data']
phoneHMMs = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()

# Add initial and final silence to each word
isolated = {}
for digit in prondict:
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']

# Construct word-level HMMs
wordHMMs = {}
for digit in isolated:
    wordHMMs[digit] = concatHMMs(phoneHMMs, isolated[digit])
