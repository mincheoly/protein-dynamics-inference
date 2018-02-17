################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Generates the XYZ and phi_psi files from the dataset

# Usage: python change_angle.py -num_clusters <num_clusters>
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
parser.add_argument('-dataset', action='store', dest='which_dataset')
args = parser.parse_args()

# Assignment arguments
which_dataset = args.which_dataset

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

# get angles
from msmbuilder.featurizer import DihedralFeaturizer
featurizer = DihedralFeaturizer(types=['phi', 'psi'], sincos=False)
print(xyz)
diheds = xyz.fit_transform_with(featurizer, 'diheds/', fmt='dir-npy')

# Compile X for phi psi
_, num_features = diheds[0].shape
pre_X = [np.reshape(traj, (traj.shape[0],num_features)) for traj in diheds]
X = np.concatenate(pre_X)

# Compile X for raw XYZ
temp = xyz[0]
_, num_atoms, num_axis = temp.xyz.shape
reference_frame = temp.slice(0, copy=True)
num_features = num_atoms*num_axis;
pre_X = [np.reshape(traj.xyz, (traj.superpose(reference_frame).xyz.shape[0],num_features)) for traj in xyz]
X_XYZ = np.concatenate(pre_X)

# determine how points to sample
if which_dataset == 'fspeptide':
	max_frame = 19000 + 1
	sampling_range = np.arange(1000, max_frame, 500)
	sampling_range = np.concatenate((np.arange(100, 1000, 100), sampling_range))
if which_dataset == 'apo_calmodulin':
	max_frame = 30000 + 1
	sampling_range = np.arange(20000, max_frame, 2000)
	sampling_range = np.concatenate((np.arange(5000, 20000, 1000), sampling_range))
	sampling_range = np.concatenate((np.arange(1000, 5000, 500), sampling_range))
	sampling_range = np.concatenate((np.arange(200, 1000, 200), sampling_range))

for num_frame in sampling_range:
	indices = np.loadtxt(folder + 'indices_'+str(num_frame) + '.csv', delimiter=',').astype(int)
	np.savetxt(folder + 'raw_phi_psi_'+str(num_frame)+'.csv', X[indices,:], delimiter=',')
	np.savetxt(folder + 'raw_XYZ_'+str(num_frame)+'.csv', X_XYZ[indices,:], delimiter=',')
	
# Also save the complete X matrix
np.savetxt(folder + 'raw_phi_psi_'+str(X.shape[0])+'.csv', X, delimiter=',')
np.savetxt(folder + 'raw_XYZ_'+str(X_XYZ.shape[0])+'.csv', X_XYZ, delimiter=',')