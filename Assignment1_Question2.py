import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

# initializing given information
p=1 #rho=1
theta=np.deg2rad(0.5) #theta 0.5 degree
x=(p*np.cos(theta), p*np.sin(theta)) #mean of normal

# empty cov matrices
sigma_x=np.zeros((2,2))
sigma_y=np.zeros((2,2))

# defining cov mat of x
sigma_x[0,0]=0.01
sigma_x[1,1]=0.005

# creating normal dist using x cov mat
samples = np.random.multivariate_normal(x, sigma_x, size=1000)

# assuming local linearity: y=f(x)=J*x
y = np.column_stack([samples[:, 0] * np.cos(samples[:, 1]-p*samples[:, 1]*np.sin(samples[:, 1])),
                     samples[:, 0] * np.sin(samples[:, 1])+p*samples[:, 1]*np.cos(samples[:, 1])])

# cov matrix for y
sigma_y[0,0]=((sigma_x[0,0])*np.cos(theta)**2)+((sigma_x[1,1])*(p**2)*(np.sin(theta)**2))
sigma_y[0,1]=(sigma_x[0,0])*np.sin(theta)*np.cos(theta)-(sigma_x[1,1])*(p**2)*np.sin(theta)*np.cos(theta)
sigma_y[1,0]=sigma_y[0,1]
sigma_y[1,1]=((sigma_x[0,0])*np.sin(theta)**2)+((sigma_x[1,1])*(p**2)*(np.cos(theta)**2))

# eigen vals and vecs for cov mat of y
eigenvalues, eigenvectors = np.linalg.eig(sigma_y)
angle=np.arctan2(eigenvectors[0][1],eigenvectors[0][0])

# plotting points and elipse
fig, ax = plt.subplots()
ax.scatter(y[:, 0], y[:, 1], marker='.', color='blue', alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Transformed Points and Elipse')
ax.grid(True)

# this is a 2 DOF chi-square problem: (x/sd_x)^2+(y/sd_y)^2=s
# from the table we can get a value for s
# I have chosen 95% certainty for data
# table: https://people.richland.edu/james/lecture/m170/tbl-chi.html
s=np.sqrt(5.991)
width = 2 * np.sqrt(eigenvalues[0])*s
height = 2 * np.sqrt(eigenvalues[1])*s

angle=np.arctan2(eigenvectors[0][1],eigenvectors[0][0])

ellipse = Ellipse((p*np.cos(theta), p*np.sin(theta)), width, height, color='red', angle=angle, alpha=0.3)
ax.add_patch(ellipse)

plt.show()
print(sigma_y)
