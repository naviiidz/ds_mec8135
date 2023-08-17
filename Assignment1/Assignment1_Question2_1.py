import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Define parameters
np.random.seed(42)
mean = np.array([1, np.radians(0.5)])  # Convert degrees to radians
cov = np.array([[0.01, 0], [0, 0.005]])

# Monte Carlo simulation
num_samples = 1000
samples = np.random.multivariate_normal(mean, cov, num_samples)

# Create the scatter plot
plt.scatter(samples[:, 0], samples[:, 1], s=10, label='Simulated Points')

# Calculate and overlay uncertainty ellipse

# this is a 2 DOF chi-square problem: (x/sd_x)^2+(y/sd_y)^2=s
# from the table we can get a value for s
# I have chosen 95% certainty for data
# table: https://people.richland.edu/james/lecture/m170/tbl-chi.html

s=np.sqrt(5.991) 
eigvals, eigvecs = np.linalg.eig(cov)
angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
ellipse = Ellipse(mean, 2 * s * np.sqrt(eigvals[0]), 2 * s * np.sqrt(eigvals[1]),
                  angle=angle, edgecolor='r', facecolor='none', label='Uncertainty Ellipse')
plt.gca().add_patch(ellipse)

# Add labels, legend, etc.
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()

# Show or save the plot
plt.show()
