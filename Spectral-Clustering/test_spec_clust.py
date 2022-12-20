import pytest
import pandas as pd
import numpy as np
import spec_clust as sc

my_data = np.loadtxt("Spectral-Clustering/spec_clust_data.csv", delimiter = ",")

#### Tests for Adjacency Matrix ####

def test_make_adj_size():
	expected = (1000,1000)
	assert sc.make_adj(my_data).shape == expected

def test_make_adj_diag():
	# Test for empty diagonal
	expected = 0
	out = sc.make_adj(my_data)
	assert np.sum(np.diag(out)) == expected

def test_make_adj_values():
	# Test that the matrix is binary
	expected = [0, 1]
	out = sc.make_adj(my_data)
	assert list(np.unique(out)) == expected

#### Tests for Laplacian ####

def test_my_laplacian_size():
	expected = (1000,1000)
	AM = sc.make_adj(my_data)
	assert sc.my_laplacian(AM).shape == expected

def test_my_laplacian_diag():
	# Test that the degree matrix diagonal is 
	#    non-negative
	expected = 1000
	AM = sc.make_adj(my_data)
	out = sc.my_laplacian(AM)
	assert np.sum(np.diag(out)>=0) == expected

def test_my_laplacian_else():
	expected = 1000*1000
	AM = sc.make_adj(my_data)
	out = sc.my_laplacian(AM)
	#Consider the top triangular half of the matrix
	upt_out = np.triu(out,1)
	#Consider the lower triangular half of the matrix
	downt_out = np.tril(out,-1)
	test_mat = upt_out + downt_out
	assert np.sum(test_mat<=0) == expected

#### Test for Kmeans ####

def test_full_kmeans():
	assert isinstance(sc.full_kmeans(my_data,6), tuple)

def test_full_kmeans_shape():
	expected = 2
	assert len(sc.full_kmeans(my_data,6)) == expected

def test_full_kmeans_labels():
	expected = 5
	label_max = np.max(sc.full_kmeans(my_data,6)[0])
	assert label_max == expected

def test_full_kmeans_center_num():
	expected = (6,2)
	centers_shape = sc.full_kmeans(my_data, 6)[1].shape
	assert centers_shape == expected

#### Spectral clustering tests ####

def test_spect_clustering():
	AM = sc.make_adj(my_data)
	L = sc.my_laplacian(AM)
	out = sc.spect_clustering(L,7)
	assert isinstance(out, tuple)

def test_spect_clustering_shape():
	expected = 2
	AM = sc.make_adj(my_data)
	L = sc.my_laplacian(AM)
	out = sc.spect_clustering(L,7)
	assert len(out) == expected

def test_spect_clustering_labels():
	expected = 6
	AM = sc.make_adj(my_data)
	L = sc.my_laplacian(AM)
	out = sc.spect_clustering(L,7)
	label_max = np.max(out[0])
	assert label_max == expected

def test_spect_clustering_center_num():
	expected = (7,7)
	AM = sc.make_adj(my_data)
	L = sc.my_laplacian(AM)
	out = sc.spect_clustering(L,7)
	centers_shape = out[1].shape
	assert centers_shape == expected


