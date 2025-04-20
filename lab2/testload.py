import numpy as np

phoneHMMs = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
print(phoneHMMs.keys())
example = np.load('lab2_example.npz', allow_pickle=True)['example'].item()
print(example.keys())
print("Digit spoken:", example['digit'])
print("LMFCC shape:", example['lmfcc'].shape)  # Number of frames × 13
print("Viterbi log-likelihood:", example['vloglik'])
print("Viterbi path:", example['vpath']) 

print(phoneHMMs['f'])
