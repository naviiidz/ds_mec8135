import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

# initializing given information
p=1 #rho=1
theta=np.deg2rad(0.5) #theta 0.5 degree
x=(p*np.cos(theta), p*np.sin(theta)) #mean of normal
values=[0.005,0.1,0.5,1]
for cov_tt in values:  
    # defining cov mat of x
    # empty cov matrices
    sigma_x=np.array([[0.01,0],[0,cov_tt]])
    sigma_y=np.zeros((2,2))

    # assuming local linearity: y=f(x)=J*x
    j=np.array([[np.cos(theta),-p*np.sin(theta)],[np.sin(theta), p*np.cos(theta)]])
    sigma_y=np.matmul(np.matmul(j,sigma_x),np.transpose(j))

    # creating normal dist using x cov mat
    samples = np.random.multivariate_normal(x, sigma_y, size=1000)

    # eigen vals and vecs for cov mat of y
    eigenvalues, eigenvectors = np.linalg.eig(sigma_y)
    angle=np.arctan2(eigenvectors[0][1],eigenvectors[0][0])

    # plotting points and elipse
    fig, ax = plt.subplots()
    ax.scatter(samples[:, 0], samples[:, 1], marker='.', color='blue', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_xlim(0.6,1.4)
    ax.set_ylim(-2.5,2.5)
    ax.set_ylabel('y')
    ax.set_title('Transformed Points and Elipse')
    ax.grid(True)

    # this is a 2 DOF chi-square problem: (x/sd_x)^2+(y/sd_y)^2=s
    # from the table we can get a value for s
    # I have chosen 95% certainty for data
    # table: https://people.richland.edu/james/lecture/m170/tbl-chi.html
    s = 5.991
    width = 2 * np.sqrt(eigenvalues[0]*s)
    height = 2 * np.sqrt(eigenvalues[1]*s)

    angle=np.arctan2(eigenvectors[0][1],eigenvectors[0][0])

    ellipse = Ellipse((p*np.cos(theta), p*np.sin(theta)), width, height, color='red', angle=angle, alpha=0.3)
    ax.add_patch(ellipse)

    plt.show()
    print("Jacobian for %.2f" %cov_tt)
    print(sigma_y)