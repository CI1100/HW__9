from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn import datasets
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(42)

iris = datasets.load_iris()
X = iris.data
y = iris.target

def visualize(X, y, pca=False, use3d=False):
    fig = plt.figure()
    if use3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    if pca:
      p = PCA(n_components=3, copy=False)
      X = p.fit_transform(X)
   
    for clas, color, marker in [(0, 'r', 'o'), (1, 'b', '*'), (2, 'g', 's')]:
        points = X[y == clas]
        if use3d:
            ax.scatter(points[:,0], points[:,1], points[:,2], s=5, marker=marker, color=color)
        else:
            ax.scatter(points[:,0], points[:,1], s=50, marker=marker, color=color)
    plt.show(block=True)
    
def find_means(X, y, n_clusters):
    means = []
    for i in range(n_clusters):
        points = X[y == i]
        means.append(np.mean(points, axis=0))
    return means

def find_distances(X, means):
    distances = []
    for mean in means:
        dist = np.linalg.norm(X - mean, axis=1)
        distances.append(dist)      
    return np.stack(distances)
    
def kmeans(X, n_clusters=3, max_iter=100):
    y = np.random.choice(np.arange(n_clusters), size=X.shape[0])
    
    for i in range(max_iter):
        means = find_means(X, y, n_clusters)
        distances = find_distances(X, means)
        new_y = np.argmin(distances, axis=0)
        if np.array_equal(new_y, y):
            print('Terminating after %d iterations.' % i)
            break
        else:
            y = new_y
    return y, means
    
def ransack(X, n_clusters, max_ransack_iter=10, max_kmeans_iter=10, max_dist=0.1):

    best_y = None
    best_dist = np.inf
    
    for i in range(max_ransack_iter):
        # Make a random split of X to maybeInliers/notInliers
        selector = np.random.choice([False, True], size=X.shape[0])
        maybeInliers = X[selector]
        notInliers = X[np.logical_not(selector)]
        # Fit kmeans
        y_km, means = kmeans(maybeInliers, n_clusters=n_clusters, max_iter=max_kmeans_iter)
        
        # Add notInliers that have distances < max_dist to newInliers
        distances = find_distances(notInliers, means)
        add_select = np.min(distances, axis=0)
        #print(add_select, distances)
        newInliers = np.concatenate([maybeInliers, notInliers[add_select <  max_dist]])
          
        # Train kmeans on newInliers
        y_km, means = kmeans(newInliers, n_clusters=n_clusters, max_iter=max_kmeans_iter)
        distances = find_distances(newInliers, means)
        new_dist = np.mean(distances)
        print('Current error: %f (best %f)' % (new_dist, best_dist))
        if new_dist < best_dist:
            best_dist = new_dist
            distances = find_distances(X, means)
            best_y = np.argmin(distances, axis=0)
    return best_y
        
#y_km = kmeans(X, n_clusters=3, max_iter=100)
#visualize(X, y_km, pca=False, use3d=False)

y_ransack = ransack(X, 3, max_dist=0.3)
visualize(X, y_ransack, pca=False, use3d=False)