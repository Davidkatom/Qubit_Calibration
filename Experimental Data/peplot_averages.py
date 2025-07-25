import os
import numpy as np
import matplotlib.pyplot as plt

# Path to the NPZ file containing measurement counts
DATA_PATH = 'IQCC_RawCounts.npz'

# Load arrays from the NPZ file
with np.load(DATA_PATH) as data:
    arrays = {k: data[k].squeeze() for k in data.files}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, (name, arr) in zip(axes, arrays.items()):
    # Average each row (subarray) of the 2D array
    averages = arr.mean(axis=1)
    ax.plot(np.arange(len(averages)), averages)
    ax.set_title(name)
    ax.set_xlabel('Index')
    ax.set_ylabel('Average')

plt.tight_layout()
plt.savefig('averages.png')
plt.show()
