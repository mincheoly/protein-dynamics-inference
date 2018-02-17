################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Generates simulations of md trajectories given a dataset using isomap
# distances

# Usage: python boot_simulation_generator.py -dataset <dataset> -num_traj <num_traj>
# -traj_length <traj_length> -n_cluster <n_cluster> -n_neighbor <n_neighbor>
# num_traj is the number of trajectories to generate
# traj_length is the length of each trajectory (number of frames)
# cluster_degree is the number of clusters that are reachable given any cluster
# frame_degree is the number of frames that are reachable given any frame
# ID is the identification of the original isomap used. Follows the format 'X_Y_Z'

# Example: python boot_sim_gen.py -dataset fspeptide -ID '30_40_97' -num_traj 50 -traj_length 50

# Outputs: md trajectory files with .xtc format

################################################################################

# imports / get data
from msmbuilder.example_datasets import FsPeptide
import numpy as np
import scipy as scipy
import argparse as ap
from msmbuilder.dataset import dataset
import mdtraj as md
import random

# Process arguments
parser = ap.ArgumentParser(description='knn simulation generator.')
parser.add_argument('-num_traj', action='store', dest='num_clusters', type=int, default=50)
parser.add_argument('-traj_length', action='store', dest='traj_length', type=int, default=50)
parser.add_argument('-dataset', action='store', dest='which_dataset')
parser.add_argument('-iso_ID', action='store', dest='iso_ID', type=str, default='30_40_97')
parser.add_argument('-cluster_degree', action='store', dest='cluster_degree', type = int, default=5)
parser.add_argument('-frame_degree', action='store', dest='frame_degree', type = int, default=5)
parser.add_argument('-random', action='store', dest='random', type=bool, default=False)
args = parser.parse_args()

# Assignment arguments
which_dataset = args.which_dataset
iso_ID = args.iso_ID
cluster_degree = args.cluster_degree
frame_degree = args.frame_degree

num_traj = args.num_clusters
traj_length = args.traj_length
sample_rand = args.random

# Obtain the raw X,Y,Z coordinates of the trajectories
X = np.load('/scratch/users/mincheol/' + which_dataset + '/raw_XYZ.dat')

# Find the dictionary
foldername = '/scratch/users/mincheol/' + which_dataset + '/dictionary/'
dict_filename = 'dict_' + which_dataset + '_' + str(cluster_degree) + '_' + str(frame_degree) + '_iso_' + iso_ID +'.json'

# Open Dictionary
import json
with open(foldername + dict_filename, 'r') as fp:
    raw_dict = json.load(fp)

# Read through the dictionary and fix all the tuples
edges  = {}
for key,value in raw_dict.items(): # not iteritems() since it's python 3.x
    edges[int(key)] = [(int(tup[0]), tup[1]) for tup in value]

# Delete all trajectories in the folder
import sys
import os
from os import listdir
from os.path import join

dir = '/scratch/users/mincheol/' + which_dataset + '/trajectories/temp/'
test=os.listdir(dir)

for item in test:
    if item.endswith('.xtc'):
        os.remove(join(dir, item))


# Generate frame orders
# The old functions do not sample smartly. Instead they just sample randomly with no bias
# in starting seed
def random_frame_old(neighbors):
    prob = []
    choice = []
    for (frame, dist) in neighbors:
        prob.append(dist)
        choice.append(frame)
    return np.random.choice(choice, p = prob)

def generate_md_traj_old(graph_dict, X, folder_name, num_traj, length, random=False):
    seed = np.linspace(0, len(graph_dict)-1, num_traj)
    for k in range(0,num_traj):
        if random:
            start = np.random.randint(0, len(graph_dict))
        else:
            start = int(seed[k])

        traj = [start]
        for i in range(0, length-1):
            neighbor = graph_dict[traj[i]]
            traj.append(random_frame(neighbor))

        our_traj = np.reshape(X[traj,:], (len(traj), len(X[0])/3, 3))
        md_traj = md.Trajectory(our_traj, md.load(fs_peptide.data_dir + '/fs-peptide.pdb').topology)
        filename = which_dataset + '_sim_' + str(cluster_degree) + '_' + str(frame_degree) + '_' + str(k+1) + '.xtc'
        md_traj.save_xtc(folder_name + filename)

# These functions are the smarter bootstrap trajectory generators
# They sample smartly (trying to sample across the whole space more quickly)
def random_frame_smart(neighbors, choices):
    prob = []
    choice = []
    for (frame, dist) in neighbors:
        if frame in choices:
            choices.remove(frame)
        prob.append(dist)
        choice.append(frame)
    return np.random.choice(choice, p = prob)

def generate_md_traj_smart(graph_dict, X, folder_name, num_traj, length):
    choices = set()
    visited = np.zeros((X.shape[0],2)) # keeps track of visited frames (frame, # of times visited)
    visited[:,0] = range(X.shape[0])
    for k in range(0,num_traj):
        if len(choices) == 0:
            idx = np.where(visited[:,1] == min(visited[:,1]))[0]
            choices = set(visited[idx,0])
        start = random.sample(choices, 1)[0]
        choices.remove(start)
        traj = [start]
        visited[start,1] += 1

        for i in range(0, length-1):
            neighbor = graph_dict[traj[i]]
            next_frame = random_frame_smart(neighbor, choices)
            traj.append(next_frame)
            visited[next_frame,1] += 1

        our_traj = np.reshape(X[traj,:], (len(traj), len(X[0])/3, 3))
        md_traj = md.Trajectory(our_traj, md.load(fs_peptide.data_dir + '/fs-peptide.pdb').topology)
        filename = which_dataset + '_sim_' + str(cluster_degree) + '_' + str(frame_degree) + '_' + str(k+1) + '.xtc'
        md_traj.save_xtc(folder_name + filename)

traj_folder = '/scratch/users/mincheol/' + which_dataset + '/trajectories/temp/'
#generate_md_traj_old(edges, X, traj_folder, num_traj, traj_length, random=sample_rand)
fs_peptide = FsPeptide()
generate_md_traj_smart(edges, X, traj_folder, num_traj, traj_length)
