################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Creates a dictionary which can be used to geerate simulations of md 
# trajectories given a dataset using isomap

# Usage: python boot_sim_dict.py -dataset <dataset> -cluster_degree <cluster_degree>
# -frame_degree <frame_degree> -n_neighbors <n_neighbors> -n_components <n_components> 
# -num_clusters <num_clusters>
# cluster_degree is the number of clusters that are reachable given any cluster
# frame_degree is the number of frames that are reachable given any frame
# n_neighbors is a number of neighbors used in the ISOMAP space
# n_components is components in the ISOMAP space
# num_clusters is the number of clusters used for clustering

# Example: python boot_sim_dict.py -dataset fspeptide

# Outputs: dictionary to create knn simulations

################################################################################

# imports / get data
from msmbuilder.example_datasets import FsPeptide
import numpy as np
import scipy as scipy
import argparse as ap
from msmbuilder.dataset import dataset

# Process arguments
parser = ap.ArgumentParser(description='knn simulation generator.')
parser.add_argument('-dataset', action='store', dest='which_dataset')
parser.add_argument('-cluster_degree', action='store', dest='cluster_degree', type = int, default=5)
parser.add_argument('-frame_degree', action='store', dest='frame_degree', type = int, default=5)
parser.add_argument('-n_neighbors', action="store", dest='n_neighbors', type=int, default=30)
parser.add_argument('-n_components', action="store", dest='n_components', type=int, default=40)
parser.add_argument('-num_clusters', action="store", dest='num_clusters', type=int, default=97)

args = parser.parse_args()

# Assignment arguments
which_dataset = args.which_dataset
n_neighbors = args.n_neighbors
n_components = args.n_components
num_clusters = args.num_clusters

cluster_degree = args.cluster_degree
frame_degree = args.frame_degree

# Combine n_neighbors and n_components to produce an ID
ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(num_clusters)

print(ID)

# Load isomap data

# Load isomap cluster labels
# Ensure that the smallest cluster label is 0!

base_filename = '/scratch/users/mincheol/' + which_dataset + '/isomap_out/'

# Load isomap frame labels
iso_label = np.load(base_filename + 'isomap_clustering_labels_' + ID + '.dat')

# Load isomap coordinate
iso_coord = np.load(base_filename + 'isomap_coordinates_' + ID + '.dat')

# Load isomap cluster centroids
iso_cent = np.load(base_filename + 'isomap_clusters_RD_' + ID + '.dat')

print(base_filename)

# create cluster adjacency matrix
cluster_num = len(np.unique(iso_label))
cluster_adj = np.zeros((cluster_num, cluster_num))

# create full adjacency matrix
for i in range(0,cluster_num-1):
    for j in range(i+1, cluster_num):
        cluster_adj[i,j] = np.linalg.norm(iso_cent[i]-iso_cent[j])

cluster_adj = cluster_adj + np.transpose(cluster_adj)

# prunce adjacency matrix
for i in range(0, cluster_num):
    ind = np.argpartition(cluster_adj[i], -cluster_num+cluster_degree+1)[-cluster_num+cluster_degree+1:]
    for j in ind:
        cluster_adj[i,j] = 0

print(cluster_adj)

# create a dictionary that describes the connections of all the frames
edges = {}
for i in range(0,cluster_num):
    cluster_frame = np.where(iso_label == i)[0] #get index of all frames in cluster i
    
    neighbor_cluster = np.where(cluster_adj[i])[0] #find connected clusters to cluster i
    neighbor_frame = np.where([x in neighbor_cluster for x in iso_label])[0] #get infex of all frames in neighboring clusters

    all_frame = np.concatenate((cluster_frame, neighbor_frame))
    length = len(all_frame)
    
    # calculate distances to all neighbors from each frame in cluster i
    for i, frame_i in enumerate(cluster_frame):
        temp = np.zeros((length))
        for j, frame_j in enumerate(all_frame): #can make this more efficient
            temp[j] = np.linalg.norm(iso_coord[frame_i]-iso_coord[frame_j]) #edjes are based on isomap distances
        temp[i] = np.nan
        # find the n nearest frames to frame_i
        if frame_degree == 0:
            n_frame = temp.shape[0]-1
            nearest = np.argpartition(temp, n_frame)[:n_frame]
        else:
            nearest = np.argpartition(temp, frame_degree)[:frame_degree]
        prob = 1/temp[nearest]
        prob = prob/sum(prob)
        neighbor = all_frame[nearest]
        neighbor = np.char.mod('%d', neighbor) # convert to string to save as json
        edges[str(frame_i)] = list(zip(neighbor, prob))

# save the dictionary
import json

foldername = '/scratch/users/mincheol/' + which_dataset + '/dictionary/'
dict_filename = 'dict_' + which_dataset + '_' + str(cluster_degree) + '_' + str(frame_degree) + '_iso_' + ID +'.json'

with open(foldername + dict_filename, 'w') as fp:
    json.dump(edges, fp)

