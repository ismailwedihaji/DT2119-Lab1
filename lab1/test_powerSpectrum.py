import numpy as np
import matplotlib.pyplot as plt
from lab1_proto import powerSpectrum
import lab1_tools  # for loading the example data

# Load example data
example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

# Get input and expected output
windowed = example['windowed']       # input to powerSpectrum()
expected_spec = example['spec']      # expected result from example

# Run your function
my_spec = powerSpectrum(windowed, nfft=512)

# Visual comparison
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.pcolormesh(expected_spec)
plt.title("Expected Power Spectrum (example['spec'])")

plt.subplot(1, 2, 2)
plt.pcolormesh(my_spec)
plt.title("Your Power Spectrum Output")
plt.tight_layout()
plt.show()

# Numerical check (optional)
mean_diff = np.mean(np.abs(expected_spec - my_spec))
max_diff = np.max(np.abs(expected_spec - my_spec))
print(f"Mean difference: {mean_diff}")
print(f"Max difference: {max_diff}")
