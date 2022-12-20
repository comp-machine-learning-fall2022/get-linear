import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import linalg as LA

from sklearn.cluster import KMeans
from scipy.spatial import distance 

'''
mak_adj takes in a numpy array array_np
then creates an adjacency matrix between the points of array_np
'''
def make_adj(array_np):
    #calculate the distance between all points
    adj_mat = distance.cdist(array_np, array_np, 'euclidean') 

    for i in range(len(adj_mat)):
        for j in range(len(adj_mat[0,:])):
            #if i and j are the same point...
            if i == j: 
                adj_mat[i,j] = 0
            #if they are too far apart...
            elif adj_mat[i,j] >= .5:
                adj_mat[i,j] = 0
            #if they are close enough...
            else:
                adj_mat[i,j] = 1

    return adj_mat

'''
my_laplacian takes in adjacency matrix A
then creates a Laplacian matrix from A
'''
def my_laplacian(adj_mat): 
    #D counts up the degree of each point by summing up 
    #the number of points that a point is next to
    D = np.diag(np.sum(adj_mat, axis = 1))
    
    #subtract the adjacency matrix from degrees matrix
    return np.subtract(D, adj_mat)

'''
spect_clustering takes in a Laplacian L and a number of eigenvectors k
then computes k-means on the eigenenvectors
'''
def spect_clustering(L, k):
    #compute eigen vectors
    w, v = LA.eig(L)

    #sort eigenvectors by their corresponding eigenvalues
    idx = np.argsort(w)
    sorted_vex = v[:, idx]

    #select first k eigenvectors
    #kmeans on eigenvectors
    return full_kmeans(sorted_vex[:, 0:k], k)

'''
full_kmeans takes in numpy array array_np and number of clusters k
then uses off the shelf kmeans from sci-kit learn
'''
def full_kmeans(array_np, k):
    km_alg = KMeans(n_clusters=k, init="random",random_state = 1, max_iter = 200)
    fit1 = km_alg.fit(array_np)
    labels = fit1.labels_
    centers = fit1.cluster_centers_
    return (labels, centers)


#load in the data
my_data = np.loadtxt("spec_clust_data.csv", delimiter = ",")

#make Laplacian
adjmat = make_adj(my_data)
L = my_laplacian(adjmat)
#Show the laplacian
#plt.imshow(L)

#run spectral clustering
labels, centers = spect_clustering(L, 2)
#show the clustering
plt.scatter(my_data[:,0], my_data[:,1], c = labels)
plt.show()


