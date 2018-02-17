################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Samples the population of MSM builder.

# Usage: python sample_msm.py -num_clusters <num_clusters>
# num_clusters is the number of clusters used for clustering

# Example: python population_sample.py -num_clusters 400 -dataset apo_calmodulin

# Outputs: 3 files
# 1) raw_XYZ - trajectory in raw format (each row is a frame)
# 3) cluster_indices - state assignment matrix for each frame in raw_XYZ

################################################################################

# imports / get data
from msmbuilder.example_datasets import FsPeptide
import numpy as np
import scipy as scipy
import argparse as ap
from msmbuilder.dataset import dataset
import mdtraj as md

# Process arguments
parser = ap.ArgumentParser(description='MSMBuilder pipeline script.')
parser.add_argument('-num_clusters', action='store', dest='num_clusters', type=int)
parser.add_argument('-dataset', action='store', dest='which_dataset')
parser.add_argument('-feature', action='store', dest='feature')
args = parser.parse_args()

# Assignment arguments
num_clusters = args.num_clusters
which_dataset = args.which_dataset
feature = args.feature

# Folder to save everything
folder = '/scratch/users/mincheol/' + which_dataset + '/sim_datasets/'

import tempfile
import os
os.chdir(tempfile.mkdtemp())

xyz = [] # placeholder
print(which_dataset)
if which_dataset == 'fspeptide':
	# Get data
	fs_peptide = FsPeptide()
	fs_peptide.cache()
	xyz = dataset(fs_peptide.data_dir + "/*.xtc",
	              topology=fs_peptide.data_dir + '/fs-peptide.pdb',
	              stride=10)
	print("{} trjaectories".format(len(xyz)))
	# msmbuilder does not keep track of units! You must keep track of your
	# data's timestep
	to_ns = 0.5
	print("with length {} ns".format(set(len(x)*to_ns for x in xyz)))

if which_dataset == 'apo_calmodulin':
	print('correct')
	xyz = dataset('/scratch/users/mincheol/apo_trajectories' + '/*.lh5', stride=10)

#featurization
from msmbuilder.featurizer import DihedralFeaturizer
featurizer = DihedralFeaturizer(types=['phi', 'psi'], sincos=False)
print(xyz)
diheds = xyz.fit_transform_with(featurizer, 'diheds/', fmt='dir-npy')

#tICA
from msmbuilder.decomposition import tICA

if which_dataset == 'fspeptide':
	tica_model = tICA(lag_time=2, n_components=4)
if which_dataset == 'apo_calmodulin':
	tica_model = tICA(lag_time=40, n_components=20)

# fit and transform can be done in seperate steps:
tica_model = diheds.fit_with(tica_model)
tica_trajs = diheds.transform_with(tica_model, 'ticas/', fmt='dir-npy')

txx = np.concatenate(tica_trajs)

# save tICA
np.savetxt(folder + 'tICA_coord_+' + which_dataset +'.csv', txx, delimiter=',')

# clustering
from msmbuilder.cluster import MiniBatchKMeans
clusterer = MiniBatchKMeans(n_clusters=num_clusters) #100 for camodulin
clustered_trajs = tica_trajs.fit_transform_with(
    clusterer, 'kmeans/', fmt='dir-npy'
)

# msm builder
from msmbuilder.msm import MarkovStateModel
from msmbuilder.utils import dump

if which_dataset == 'fspeptide':
	msm = MarkovStateModel(lag_time=2, n_timescales=20, ergodic_cutoff='on')
if which_dataset == 'apo_calmodulin':
	msm = MarkovStateModel(lag_time=20, n_timescales=20, ergodic_cutoff='on')

msm.fit(clustered_trajs)

# Concatenate the trajectories in cluster indices
cluster_indices = np.concatenate(clustered_trajs)

# Compile X
if feature == 'XYZ':
	temp = xyz[0]
	_, num_atoms, num_axis = temp.xyz.shape
	reference_frame = temp.slice(0, copy=True)
	num_features = num_atoms*num_axis;
	pre_X = [np.reshape(traj.xyz, (traj.superpose(reference_frame).xyz.shape[0],num_features)) for traj in xyz]
	X = np.concatenate(pre_X)
if feature == 'angle':
	_, num_features = diheds[0].shape
	pre_X = [np.reshape(traj, (traj.shape[0],num_features)) for traj in diheds]
	X = np.concatenate(pre_X)

# save MSM
import msmbuilder.utils as msmutils
msmutils.dump(msm, '/scratch/users/mincheol/' + which_dataset + '/sim_datasets/msm_' + which_dataset + '.pkl')

# save assignments
np.savetxt(folder + 'msm_clustering_labels.csv', cluster_indices, delimiter=',')

# Generate sampled data
# Find frame limit

limit_list = []
state_list = []
for state in msm.mapping_.keys():
    num_frame = np.where(cluster_indices == state)[0].shape[0]
    prob = msm.populations_[msm.mapping_[state]]
    limit = num_frame/prob
    limit_list.append(limit)
    state_list.append(state)

limiting_state = state_list[np.argmin(limit_list)] #original frame label
max_frame = int(limit_list[msm.mapping_[limiting_state]])
print('Max frames: ')
print(max_frame)

# determine how points to sample
if which_dataset == 'fspeptide':
	sampling_range = np.arange(1000, max_frame, 500)
	sampling_range = np.concatenate((np.arange(100, 1000, 100), sampling_range))
if which_dataset == 'apo_calmodulin':
	sampling_range = np.arange(20000, max_frame, 2000)
	sampling_range = np.concatenate((np.arange(5000, 20000, 1000), sampling_range))
	sampling_range = np.concatenate((np.arange(1000, 5000, 500), sampling_range))
	sampling_range = np.concatenate((np.arange(200, 1000, 200), sampling_range))

for num_frame in sampling_range:

	# If the number of frames is too large continue
	if num_frame > max_frame:
		print('too large: ')
		print(max_frame)
		continue

	# Number of frames to sample from each state
	num_state_frames = np.array(num_frame*msm.populations_).astype(int)

	# Go through each state and take the appropriate number of frames
	frame_idx = np.empty((0,))
	for state in msm.mapping_.keys():
	    options = np.where(cluster_indices == state)[0]
	    frame_idx = np.hstack((frame_idx,np.random.choice(options, num_state_frames[msm.mapping_[state]], replace=False)))
	frame_idx = frame_idx.astype(int)

	# Cluster label for each index
	label_hat = np.repeat(list(msm.mapping_.keys()), num_state_frames)

	# Save data
	X_hat = X[frame_idx, :]
	np.savetxt(folder + 'raw_'+feature+'_'+str(num_frame)+'.csv', X_hat, delimiter=',')
	np.savetxt(folder + 'indices_'+str(num_frame)+'.csv', frame_idx, delimiter=',')
	np.savetxt(folder + 'sample_cluster_assignment_'+str(num_frame)+'.csv', label_hat, delimiter=',')

# Also save the complete X matrix
np.savetxt(folder + 'raw_'+feature+'_'+str(X.shape[0])+'.csv', X, delimiter=',')