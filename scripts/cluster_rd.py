################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Clusters in the reduced dimension, wether PC or raw ISOMAP components

# Usage: python cluster_rd.py -method <method> -n_neighbors <n_neighbors> -n_components <n_components> -dataset <dataset label> -sample_rate <sampling rate> -pc <pc> -n_clusters <n_clusters>
# method is the method to use. Currently: KernelPCA, Isomap, and SpectralEmbedding
# n_neighbors is a number of neighbors to use
# n_components is components in the reduced dimension
# dataset is which dataset to use
# sample_rate is the subsampling rate
# pc is the number of principal components you want. This would be 0 for raw reduced dimension.
# n_clusters is the number of clusters

# Example: python cluster_rd.py -method isomap -n_neighbors 30 -n_components 40 -dataset fspeptide -sample_rate 1.0 -pc 4 -n_clusters 97

# Output: A numpy dat file containing the clustering
# These files are saved with a X_pc_<method>_<n_neighbors>_<n_components>_<sample_rate>_.dat naming scheme.

################################################################################

# imports
import numpy as np
import argparse as ap
from sklearn.cluster import KMeans

# Set up ArgumentParser
parser = ap.ArgumentParser(description='Apply dimension reduction script.')
parser.add_argument('-method', action="store", dest='method')
parser.add_argument('-n_components', action="store", dest='n_components', type=int)
parser.add_argument('-n_neighbors', action="store", dest='n_neighbors', type=int)
parser.add_argument('-dataset', action='store', dest='which_dataset')
parser.add_argument('-sample_rate', action='store', dest='sample_rate', type=float)
parser.add_argument('-pc', action="store", dest='pc', type=int)
parser.add_argument('-n_clusters', action='store', dest='n_clusters', type=int)
args = parser.parse_args()

# Assign argument variables
method = args.method
n_neighbors = args.n_neighbors
n_components = args.n_components
which_dataset = args.which_dataset
sample_rate = args.sample_rate
pc = args.pc
n_clusters = args.n_clusters

# Combine n_neighbors and n_components to produce an ID
ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(sample_rate)

# Determine if PCA space or raw reduced dimension space
if pc != 0:
	X_pc = np.load('/scratch/users/mincheol/' + which_dataset + '/principal_components/X_pc_'+ method +'_' + ID + '_' + str(pc) + '.dat')
else:
	X_pc = np.load('/scratch/users/mincheol/' + which_dataset + '/reduced_dimension/X_'+ method +'_' + ID + '.dat')



# use K means to cluster and save data
kmeans = KMeans(n_clusters=n_clusters).fit(X_pc)
print('Done clustering.')

# save centroids in reduced dimension
if pc != 0:
	kmeans.cluster_centers_.dump('/scratch/users/mincheol/' + which_dataset + '/clusters/centers_pc_'+ method +'_' + ID + '_' + str(pc) + '.dat')
else:
	kmeans.cluster_centers_.dump('/scratch/users/mincheol/' + which_dataset + '/clusters/centers_'+ method +'_' + ID + '.dat')
print("Cluster centers saved in reduced dimension")

# save assignments
if pc != 0:
	kmeans.labels_.dump('/scratch/users/mincheol/' + which_dataset + '/clusters/labels_pc_'+ method +'_' + ID + '_' + str(pc) + '.dat')
else:
	kmeans.labels_.dump('/scratch/users/mincheol/' + which_dataset + '/clusters/labels_'+ method +'_' + ID + '.dat')
print("Clusters assignments saved")

