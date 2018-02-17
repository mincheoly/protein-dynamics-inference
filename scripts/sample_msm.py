################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Generates trajectories from MSM builder.

# Usage: python sample_msm.py -num_clusters <num_clusters>
# num_clusters is the number of clusters used for clustering

# Example: python sample_msm.py -num_clusters 400 -dataset apo_calmodulin

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
args = parser.parse_args()

# Assignment arguments
num_clusters = args.num_clusters
which_dataset = args.which_dataset

import tempfile
import os
os.chdir(tempfile.mkdtemp())

xyz = [] # placeholder
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
featurizer = DihedralFeaturizer(types=['phi', 'psi'])
print(xyz)
print(which_dataset)
diheds = xyz.fit_transform_with(featurizer, 'diheds/', fmt='dir-npy')  #?????????????????????????????

#tICA
from msmbuilder.decomposition import tICA
tica_model = tICA(lag_time=2, n_components=4)
# fit and transform can be done in seperate steps:
tica_model = diheds.fit_with(tica_model)
tica_trajs = diheds.transform_with(tica_model, 'ticas/', fmt='dir-npy')

txx = np.concatenate(tica_trajs)

# clustering
from msmbuilder.cluster import MiniBatchKMeans
clusterer = MiniBatchKMeans(n_clusters=num_clusters)
clustered_trajs = tica_trajs.fit_transform_with(
    clusterer, 'kmeans/', fmt='dir-npy'
)

# msm builder
from msmbuilder.msm import MarkovStateModel
from msmbuilder.utils import dump
msm = MarkovStateModel(lag_time=2, n_timescales=20, ergodic_cutoff='off')
msm.fit(clustered_trajs)

# Get MFPT
from msmbuilder.tpt import mfpts 
mfpt_matrix = mfpts(msm)

# Get flux matrix
Pi = np.diag(msm.populations_)
Pi = scipy.linalg.fractional_matrix_power(Pi, 1)
Pi_L = scipy.linalg.fractional_matrix_power(Pi, 0.5)
Pi_R = scipy.linalg.fractional_matrix_power(Pi, -0.5)
T = msm.transmat_
flux = np.linalg.multi_dot([Pi_L,T,Pi_R])

# combine all trajectories into a trajectory "bag"
frames_bag = []
for idx, trajectories in enumerate(xyz):
    if idx == 0:
        frames_bag = trajectories
    if idx != 0:
        frames_bag = frames_bag.join(trajectories)
num_frames, num_atoms, num_axis = frames_bag.xyz.shape

# Concatenate the trajectories in cluster indices
cluster_indices = np.concatenate(clustered_trajs)

# compute XYZ coordinates of cluster centers
cluster_centers = np.empty((num_clusters, num_atoms*num_axis), dtype=float)
X = np.reshape(frames_bag.xyz, (num_frames, num_atoms*num_axis))
for idx in range(num_clusters):
    indices = (cluster_indices == idx)
    cluster_centers[idx, :] = X[indices,:].mean(axis=0)

# Get MFPT
from msmbuilder.tpt import mfpts 
mfpt_matrix = mfpts(msm)

# Get flux matrix
Pi = np.diag(msm.populations_)
Pi = scipy.linalg.fractional_matrix_power(Pi, 1)
Pi_L = scipy.linalg.fractional_matrix_power(Pi, 0.5)
Pi_R = scipy.linalg.fractional_matrix_power(Pi, -0.5)
T = msm.transmat_
flux = np.linalg.multi_dot([Pi_L,T,Pi_R])

# save MFPT matrix
mfpt_matrix.dump('/scratch/users/mincheol/' + which_dataset + '/sim_datasets/msm_mfpt_mat.dat')

# save clusters
cluster_centers.dump('/scratch/users/mincheol/' + which_dataset + '/sim_datasets/msm_clusters_XYZ.dat')	

# save assignments
cluster_indices.dump('/scratch/users/mincheol/' + which_dataset + '/sim_datasets/msm_clustering_labels.dat')

# save flux matrix
flux.dump('/scratch/users/mincheol/' + which_dataset + '/sim_datasets/msm_flux_mat.dat')

# save population vector
msm.populations_.dump('/scratch/users/mincheol/' + which_dataset + '/sim_datasets/msm_pop_vec.dat')

# Generate trajectoies
for stride_eff in np.arange(250, 1010, 10):
	# get length of the trajectory
	filename = '/scratch/users/mincheol/apo_calmodulin/datasets/indices_' + str(stride_eff) + '.dat'
	indices = np.load(filename)
	traj_length = indices.shape[0]

	# generate and populate
	traj_raw = msm.sample(msm.transmat_, traj_length) #Raw trajectory with only state labels
	traj = []
	for state in traj_raw:
	    options = np.where(cluster_indices == state)[0]
	    next_frame = np.random.choice(options)
	    traj.append(next_frame)
	traj = np.array(traj)
	    
	folder = '/scratch/users/mincheol/'+which_dataset+'/sim_datasets/'
	X[traj,:].dump(folder+'raw_XYZ_'+str(stride_eff)+'.dat') # Saves a .dat file
	traj.dump(folder+'indices_'+ str(stride_eff) +'.dat')