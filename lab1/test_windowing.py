import numpy as np
import matplotlib.pyplot as plt
from lab1_proto import enframe, preemp, windowing

# Load example data
example = np.load("lab1_example.npz", allow_pickle=True)["example"].item()

# Step-by-step processing
samples = example["samples"]
frames = enframe(samples, winlen=400, winshift=200)
preemph = preemp(frames, p=0.97)
my_windowed = windowing(preemph)

# Expected windowed result from example
expected_windowed = example["windowed"]

# Visual comparison
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.pcolormesh(expected_windowed)
plt.title("Expected Windowed Output")

plt.subplot(1, 2, 2)
plt.pcolormesh(my_windowed)
plt.title("Your Windowing Output")

plt.tight_layout()
plt.show()

# Optional: print small difference
diff = np.abs(my_windowed - expected_windowed)
print("Mean difference:", np.mean(diff))
print("Max difference:", np.max(diff))
