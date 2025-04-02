import numpy as np
import matplotlib.pyplot as plt
from lab1_proto import enframe, preemp

# Load example data
example = np.load("lab1_example.npz", allow_pickle=True)["example"].item()
samples = example["samples"]
expected_preemp = example["preemph"]

# First, create frames like in the example
frames = enframe(samples, winlen=400, winshift=200)

# Now apply your preemp() function
my_preemp = preemp(frames, p=0.97)

# Compare using visualization
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.pcolormesh(expected_preemp)
plt.title("Expected Pre-emphasis")

plt.subplot(1, 2, 2)
plt.pcolormesh(my_preemp)
plt.title("Actual Pre-emphasis Output")

plt.tight_layout()
plt.show()

# Also check numeric similarity
diff = np.abs(my_preemp - expected_preemp)
print("Mean difference:", np.mean(diff))

