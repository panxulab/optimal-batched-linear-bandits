import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Define the vectors in R^2
vectors = np.array([[1, 0], [0, 1], [0.9, 0.2]])

# Create the plot
fig, ax = plt.subplots()

# Plot each vector using quiver
for vector in vectors:
    ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='black')

# Define the region boundaries for shading
region1_x = np.linspace(0, 1.1, 100)
region1_y1 = 0.5 * region1_x  # y = x/2 for x=2y
region1_y2 = 9 * region1_x / 8  # y = 9x/8 for 9x=8y

# Fill the regions with different colors
ax.fill_between(region1_x, 0, region1_y1, color='lightblue', alpha=0.5)  # Region below x=2y
ax.fill_between(region1_x, region1_y1, region1_y2, where=(region1_y1 < region1_y2), color='lightgreen', alpha=0.5)  # Between x=2y and 9x=8y
ax.fill_between(region1_x, region1_y2, 1.1, color='salmon', alpha=0.5)  # Above 9x=8y

# Draw the specific dashed lines x=2y for x>0 and 9x=8y
ax.plot(region1_x, region1_y1, 'k--')  # Line for x=2y, x>0
ax.plot(region1_x, region1_y2, 'k--')  # Line for 9x=8y

# Add an elongated and slightly tilted ellipse in the bottom right region
ellipse = Ellipse(xy=(0.95, 0.3), width=0.1, height=0.5, angle=-10, edgecolor='black', facecolor='orange', alpha=0.5)
ax.add_patch(ellipse)

# Set the axes limits
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

# Set the aspect of the plot to be equal
ax.set_aspect('equal')

# Annotate the vectors
ax.text(0.95, 0.35, r'$\hat\theta$', ha='left',fontsize='large')
ax.plot(0.95, 0.3, 'ko')
ax.text(1.05, 0, r'$x_1=(1,0)$', ha='left')
ax.text(0.05, 0.95, r'$x_3=(0,1)$', ha='left')
ax.text(0.7, 0.2, r'$x_2=(1-\epsilon, 2\epsilon)$', ha='left')

ax.text(0.5, 0.15, r'$C_1$', ha='left',fontsize='large')
ax.text(0.6, 0.5, r'$C_2$', ha='left',fontsize='large')
ax.text(0.25, 0.75, r'$C_3$', ha='left',fontsize='large')
# Annotate the lines with their equations
# ax.text(1.1, 0.55, 'x=2y', va='bottom', ha='right')
# ax.text(1.1, 9 * 1.1 / 8, '9x=8y', va='bottom', ha='right')

# Turn off the axes
ax.axis('off')

# Show the plot
plt.show()
