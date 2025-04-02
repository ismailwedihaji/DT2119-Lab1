import numpy as np
import matplotlib.pyplot as plt
from lab1_proto import enframe
import lab1_tools

# Load the example dictionary
example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()

# Get raw samples and correct output for frames
samples = example['samples']
expected_frames = example['frames']

# Run your enframe function
my_frames = enframe(samples, winlen=400, winshift=200)

# Visualize both (your result vs expected)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.pcolormesh(expected_frames)
plt.title('Expected (example["frames"])')

plt.subplot(1, 2, 2)
plt.pcolormesh(my_frames)
plt.title('Actual enframe() result')

plt.tight_layout()
plt.show()

diff = np.abs(expected_frames - my_frames)
print("Mean difference:", np.mean(diff))