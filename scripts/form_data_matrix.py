################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Generates the raw X matrix (rows are frames, cols are XYZ-XYZ-... coordinates)

# Usage: python form_data_matrix.py -dataset <dataset label>
# dataset label is which dataset to use

# Example: python form_data_matrix.py -dataset calmodulin

# Output: 1 file
# 1) X matrix, matrix form of the raw frames (frame bag)


################################################################################

# get data
import numpy as np
import argparse as ap
import mdtraj as md
from sklearn import manifold
from msmbuilder.example_datasets import FsPeptide
import sys
fs_peptide = FsPeptide()
fs_peptide.cache()

# additional imports
import tempfile
import os
os.chdir(tempfile.mkdtemp())

# Set up ArgumentParser
parser = ap.ArgumentParser(description='X matrix forming script.')
parser.add_argument('-dataset', action='store', dest='which_dataset')
parser.add_argument('-stride', action='store', dest='stride', type=int)
args = parser.parse_args()

# Assign argument variables
which_dataset = args.which_dataset
stride = args.stride

# Compile all trajectories
from msmbuilder.dataset import dataset
xyz = [] # placeholder
if which_dataset == 'fspeptide':
	# Get data
	fs_peptide = FsPeptide()
	fs_peptide.cache()
	xyz = dataset(fs_peptide.data_dir + "/*.xtc", topology=fs_peptide.data_dir + '/fs-peptide.pdb',stride=stride)
if which_dataset == 'apo_calmodulin':
	xyz = dataset('/scratch/users/mincheol/apo_trajectories' + '/*.lh5', stride=stride)
if which_dataset == 'holo_calmodulin':
	xyz = dataset('/scratch/users/mincheol/holo_trajectories' + '/*.lh5',stride=stride)

# Combine all trajectories into a trajectory "bag"
temp = xyz[0]
_, num_atoms, num_axis = temp.xyz.shape
reference_frame = temp.slice(0, copy=True)
num_features = num_atoms*num_axis;
pre_X = [np.reshape(traj.xyz, (traj.superpose(reference_frame).xyz.shape[0],num_features)) for traj in xyz]
X = np.concatenate(pre_X)
print(which_dataset)
print('Original shape of X')
print(X.shape)

more_strides = [i*10 for i in range(61, 101)]
for more_stride in more_strides:
	idx = [i for i in range(X.shape[0])]
	sampled_idx = idx[::int(more_stride/10)]
	np.array(sampled_idx).dump('/scratch/users/mincheol/' + which_dataset +'/datasets/indices_' + str(more_stride) + '.dat')
	sampled_X = X[sampled_idx, :]
	sampled_X.dump('/scratch/users/mincheol/' + which_dataset +'/datasets/raw_XYZ_' + str(more_stride) + '.dat')