import numpy as np

# Step 3.1 - Load phoneme HMMs
phoneHMMs = np.load("lab2_models_onespkr.npz", allow_pickle=True)["phoneHMMs"].item()

# List all phoneme model names
print("Available phonemes:", sorted(phoneHMMs.keys()))
print()

# Inspect one phoneme model structure (e.g., 'ah')
phoneme_name = 'ah'
model = phoneHMMs[phoneme_name]
print(f"Phoneme: {phoneme_name}")
print("Startprob shape:", model['startprob'].shape)
print("Transmat shape:", model['transmat'].shape)
print("Means shape:", model['means'].shape)
print("Covars shape:", model['covars'].shape)
print("Startprob vector:", model['startprob'])
print("Transmat matrix:\n", model['transmat'])
print()

# Step 3.2 - Build isolated word HMMs from phoneme sequences
from prondict import prondict
from lab2_proto import concatHMMs

# Add silence before and after
isolated = {digit: ['sil'] + phones + ['sil'] for digit, phones in prondict.items()}

# Concatenate phoneme HMMs into word-level HMMs
wordHMMs = {}
for digit, phonemes in isolated.items():
    wordHMMs[digit] = concatHMMs(phoneHMMs, phonemes)

# Inspect structure of a few digit models
for digit in ['1', '4', '7']:
    hmm = wordHMMs[digit]
    print(f"Digit '{digit}' model:")
    print(" Phonemes:", isolated[digit])
    print(" Emitting states:", hmm['means'].shape[0])
    print(" Startprob shape:", hmm['startprob'].shape)
    print(" Transmat shape:", hmm['transmat'].shape)
    print()
