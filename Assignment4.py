import numpy as np
import random
import matplotlib.pyplot as plt

threshold=0.005

source = np.array([ [1.90659, 2.51737], [2.20896, 1.1542], 
                    [2.37878, 2.15422], [1.98784, 1.44557], 
                    [2.83467, 3.41243], [9.12775, 8.60163], 
                    [4.31247, 5.57856], [6.50957, 5.65667], 
                    [3.20486, 2.67803], [6.60663, 3.80709], 
                    [8.40191, 3.41115], [2.41345, 5.71343], 
                    [1.04413, 5.29942], [3.68784, 3.54342], 
                    [1.41243, 2.6001]])

destination = np.array([[5.0513, 1.14083], [1.61414, 0.92223], 
                        [1.95854, 1.05193], [1.62637, 0.93347], 
                        [2.4199, 1.22036], [5.58934, 3.60356], 
                        [3.18642, 1.48918], [3.42369, 1.54875], 
                        [3.65167, 3.73654], [3.09629, 1.41874], 
                        [5.55153, 1.73183], [2.94418, 1.43583], 
                        [6.8175, 0.01906], [2.62637, 1.28191], 
                        [1.78841, 1.0149]])


def search_inliers(source, destination, threshold):
    n_points = source.shape[0]
    best_inliers = []
    for j in range(10):  
        #getting four random points from all sources
        indices = random.sample(range(n_points), 4) 
        #2n*9 matrix of homography
        A = []
        for i in indices:
            x, y = source[i]
            x_p, y_p = destination[i]
            A.append([x, y, 1, 0, 0, 0, -x * x_p, -y * x_p, -x_p])
            A.append([0, 0, 0, x, y, 1, -x * y_p, -y * y_p, -y_p])
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        H = V[-1, :].reshape(3, 3)
        errors_mat = []
        for i in range(n_points):
            X = source[i]
            T = destination[i]
            point = np.array([X[0], X[1], 1]).reshape(3, 1)
            p_projected = np.matmul(H , point)
            p_projected /= p_projected[2, 0]
            error = np.sqrt((p_projected[0, 0] - T[0]) ** 2 + (p_projected[1, 0] - T[1]) ** 2)
            errors_mat.append(error)   
        inliers=[]
        for i in range(len(errors_mat)):
            if errors_mat[i] < threshold:
                inliers.append(i)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    return best_inliers



def homography(source, destination, inliers):
    num_inliers = len(inliers)
    A = np.zeros((2 * num_inliers, 9))
    for i in range(len(inliers)):
        inlier=inliers[i]
        x, y = source[inlier]
        x_p, y_p = destination[inlier]
        A[2*i] = [x, y, 1, 0, 0, 0, -x*-x_p, -y*x_p, -x_p]
        A[2*i+1] = [0, 0, 0, x, y, 1, -x*y_p, -y*y_p, -y_p]
    U, S, V = np.linalg.svd(A)
    homography_mat = V[-1,:].reshape(3, 3)
    return homography_mat

def plot_data(source, destination):
    n_points = source.shape[0]    
    plt.scatter(source[:, 0], source[:, 1], color='r', marker='*', label='Source')
    plt.scatter(destination[:, 0], destination[:, 1], color='b', marker='*', label='Destination')
    # Connect inliers with lines
    for i in range(n_points):
        plt.plot([source[i, 0], destination[i, 0]], [source[i, 1], destination[i, 1]], 'c--')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original Points and Mapping')
    plt.show()

def plot_inliers(source, destination, inliers):
    n_points = source.shape[0]    
    outliers=list(set(range(n_points))-set(inliers))
    plt.scatter(source[outliers, 0], source[outliers, 1], color='b', marker='x', label='Outlier points')
    plt.scatter(destination[outliers, 0], destination[outliers, 1], color='r', marker='x')
    plt.scatter(source[inliers, 0], source[inliers, 1], marker='o', label='Inlier points')
    plt.scatter(destination[inliers, 0], destination[inliers, 1], marker='o')
    # Connect inliers with lines
    for i in inliers:
        plt.plot([source[i, 0], destination[i, 0]], [source[i, 1], destination[i, 1]], 'c--')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Inliers & Outliers')
    plt.show()

def plot_outliers(source, destination, inliers):
    n_points = source.shape[0]    
    outliers=list(set(range(n_points))-set(inliers))
    plt.scatter(source[outliers, 0], source[outliers, 1], color='b', marker='x', label='Outlier points')
    plt.scatter(destination[outliers, 0], destination[outliers, 1], color='r', marker='x')
    plt.scatter(source[inliers, 0], source[inliers, 1], marker='o', label='Inlier points')
    plt.scatter(destination[inliers, 0], destination[inliers, 1], marker='o')
    # Connect inliers with lines
    for i in outliers:
        plt.plot([source[i, 0], destination[i, 0]], [source[i, 1], destination[i, 1]], 'r--')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Outlier Transformations')
    plt.show()

if __name__=="__main__":
    #plot_data(source, destination)
    inliers = search_inliers(source, destination, threshold)
    homography = homography(source, destination, inliers)
    plot_inliers(source, destination, inliers)
    #plot_outliers(source, destination, inliers)