import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Define parameters
np.random.seed(42)

rho=1
theta=np.radians(0.5)

# List of x values and covariance matrices
x_values = [
    np.array([1, np.radians(0.5)]),
    np.array([1, np.radians(0.5)]),
    np.array([1, np.radians(0.5)]),
    np.array([1, np.radians(0.5)])
]

cov_matrices = [
    np.array([[0.01, 0], [0, 0.005]]),
    np.array([[0.01, 0], [0, 0.1]]),
    np.array([[0.01, 0], [0, 0.5]]),
    np.array([[0.01, 0], [0, 1]])
]

# Simulate and plot for each x-covariance pair
for x, cov in zip(x_values, cov_matrices):
    # Sample 1000 points
    num_samples = 1000
    samples_polar = np.random.multivariate_normal(x, cov, num_samples)

    # Convert polar coordinates to Cartesian coordinates
    samples_cartesian = np.column_stack((samples_polar[:, 0] * np.cos(samples_polar[:, 1]),
                                         samples_polar[:, 0] * np.sin(samples_polar[:, 1])))
    
    j=np.array([[np.cos(theta),-rho*np.sin(theta)]
                ,[np.sin(theta), rho*np.cos(theta)]])
    sigma_y=np.matmul(np.matmul(j,cov),np.transpose(j))

    # Calculate uncertainty ellipse parameters
    s = np.sqrt(5.991)
    eigvals, eigvecs = np.linalg.eig(sigma_y)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Create the scatter plot
    plt.scatter(samples_cartesian[:, 0], samples_cartesian[:, 1], s=10, label='Transformed Points')

    # Overlay uncertainty ellipse
    ellipse = Ellipse(x, 2 * s * np.sqrt(eigvals[0]), 2 * s * np.sqrt(eigvals[1]),
                      angle=angle, edgecolor='r', facecolor='none', label='Uncertainty Ellipse')
    plt.gca().add_patch(ellipse)

    # Add labels, legend, etc.
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    # Show or save the plot
    plt.show()
