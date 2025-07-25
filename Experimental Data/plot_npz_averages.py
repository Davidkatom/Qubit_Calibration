import numpy as np
import matplotlib.pyplot as plt

# Use a non-interactive backend to allow running without a display
plt.switch_backend('Agg')

# Load NPZ file containing 2D arrays
npz_path = 'Experimental Data/IQCC_RawCounts.npz'
data = np.load(npz_path)

# Arrays expected in the file
array_names = ['I_plus_x', 'I_minus_x', 'I_plus_y', 'I_minus_y']

# Prepare 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for ax, name in zip(axes, array_names):
    # Each array has shape (1, N, M) so remove the extra dimension
    arr = data[name].squeeze()
    # Average each subarray (over rows)
    means = arr.mean(axis=0)
    # Plot the means
    ax.plot(range(len(means)), means, marker='o')
    ax.set_title(name)
    ax.set_xlabel('Index')
    ax.set_ylabel('Average')

plt.tight_layout()

# Save figure to file
plt.savefig('npz_averages.png')
