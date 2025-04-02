# # import numpy as np

# # # Load the file
# # data = np.load("lab1_data.npz", allow_pickle=True)["data"]

# # # Preview the first item
# # print("Type:", type(data))
# # print("Number of utterances:", len(data))
# # print("Keys in first item:", data[0].keys())
# # print("Sample data (first 100 values):", data[0]['samples'][:100])


# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data
# data = np.load('lab1_data.npz', allow_pickle=True)['data']

# # Pick the first utterance
# utterance = data[0]

# # Print what keys are inside
# print(utterance.keys())  # to show you all info

# # Plot the waveform (the sound)
# plt.plot(utterance['samples'])
# plt.title(f"Digit: {utterance['digit']}, Speaker: {utterance['speaker']}")
# plt.xlabel("Sample index")
# plt.ylabel("Amplitude")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
# Load the example utterance
example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
print(example.keys())


samples = example['samples']

plt.plot(samples)
plt.title('Raw speech signal (samples)')
plt.xlabel('Sample index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

print("Sampling rate:", example['samplingrate'])
print("Length of samples:", len(samples))
