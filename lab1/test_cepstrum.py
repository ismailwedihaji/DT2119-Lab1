import numpy as np
import matplotlib.pyplot as plt
from lab1_proto import cepstrum
from lab1_tools import lifter

# Load the example dictionary
example = np.load("lab1_example.npz", allow_pickle=True)['example'].item()

# Step 1: Get input and reference data
logmel = example['mspec']           # input to cepstrum()
expected_mfcc = example['mfcc']     # DCT only (no liftering)
expected_lmfcc = example['lmfcc']   # DCT + liftering

# Step 2: Apply cepstrum and liftering
my_mfcc = cepstrum(logmel, nceps=13)
my_lmfcc = lifter(my_mfcc, lifter=22)

# Step 3: Visualize and compare
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.pcolormesh(expected_mfcc)
plt.title("Expected MFCC (no lifter)")

plt.subplot(2, 2, 2)
plt.pcolormesh(my_mfcc)
plt.title("Actual MFCC (no lifter)")

plt.subplot(2, 2, 3)
plt.pcolormesh(expected_lmfcc)
plt.title("Expected Liftered MFCC")

plt.subplot(2, 2, 4)
plt.pcolormesh(my_lmfcc)
plt.title("Actual Liftered MFCC")

plt.tight_layout()
plt.show()

# Step 4: Numeric check
print("Mean diff (MFCC):", np.mean(np.abs(my_mfcc - expected_mfcc)))
