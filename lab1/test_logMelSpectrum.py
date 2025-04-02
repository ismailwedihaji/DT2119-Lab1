import numpy as np
import matplotlib.pyplot as plt
from lab1_proto import logMelSpectrum

# Load the example data
example = np.load("lab1_example.npz", allow_pickle=True)["example"].item()

# Get the power spectrum and expected Mel spectrum
power_spec = example["spec"]               # Input to your function
expected_mspec = example["mspec"]          # Correct output

# Call your function
my_mspec = logMelSpectrum(power_spec, samplingrate=20000)

# Compare and visualize
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.pcolormesh(expected_mspec)
plt.title("Expected Mel Spectrum")

plt.subplot(1, 2, 2)
plt.pcolormesh(my_mspec)
plt.title("Actual Mel Spectrum")

plt.tight_layout()
plt.show()

# Optional: print numeric difference
diff = np.abs(expected_mspec - my_mspec)
print("Mean difference:", np.mean(diff))

