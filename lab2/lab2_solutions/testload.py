import numpy as np

# Load models
phoneHMMs_single = np.load('../lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
phoneHMMs_all = np.load('../lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
example = np.load('../lab2_example.npz', allow_pickle=True)['example'].item()


# Print phoneme keys (should be the same)
print("Phoneme keys:\n", phoneHMMs_single.keys())

# Compare the 'ow' phoneme
print("\n--- 'ow' phoneme comparison ---")
print("Single speaker 'ow' mean:\n", phoneHMMs_single['ow']['means'])
print("\nMulti speaker 'ow' mean:\n", phoneHMMs_all['ow']['means'])

# Optional: Compare one value difference
print("\nAre means identical?", np.allclose(phoneHMMs_single['ow']['means'], phoneHMMs_all['ow']['means']))
