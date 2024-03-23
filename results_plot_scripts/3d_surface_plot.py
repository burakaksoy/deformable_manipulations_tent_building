#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Given data
d_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1])
h_values = np.array([0.3, 0.5, 0.7, 0.9])
success_rates = np.array([
    [0.8,	0.9,	1.0,	0.85,	0.9,	0.9],
    [0.45,	0.3,	0.4,	0.5,	0.55,	0.55],
    [0.3,	0.35,	0.25,	0.6,	0.4,	0.5],
    [0.05,	0.25,	0.2,	0.6,	0.4,	0.4]
])


# Create a meshgrid
D, H = np.meshgrid(d_values, h_values)

# Plotting with equal scales for distance and height
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Create a surface plot
surf = ax.plot_surface(D, H, success_rates, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('Distance (d)')
ax.set_ylabel('Height (h)')
ax.set_zlabel('Success Rate')
ax.set_title('3D Surface Plot of Success Rates')

# Setting the same scale for distance and height
ax.set_box_aspect([1,1,1])  # Equal aspect ratio

# Adjusting the position of the colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)

# Show the plot
plt.show()