################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Runs MSMBuilder in a given dataset. Outputs various results we
# want from the MSM Builder module

# Usage: python msmbuilder_pipeline.py -num_clusters <num_clusters>
# num_clusters is the number of clusters used for clustering

# Example: python msmbuilder_pipeline.py -num_clusters 400 -dataset apo_calmodulin

# Outputs: 3 files
# 1) MFPT matrix (num_state x num_state) - mean first passage time matrix from/to each state
# 2) msm_cluster_XYZ - Centroid of each state cluster in XYZ coordinates
# 3) cluster_indices - state assignment matrix for each frame

################################################################################

# imports / get data
from msmbuilder.example_datasets import FsPeptide
import numpy as np
import scipy as scipy
import argparse as ap
import msmbuilder.utils as utils
from msmbuilder.dataset import dataset

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
	              stride=1)

if which_dataset == 'apo_calmodulin':
	xyz = dataset('/scratch/users/mincheol/apo_trajectories' + '/*.lh5', stride=10)

#featurization
from msmbuilder.featurizer import DihedralFeaturizer
featurizer = DihedralFeaturizer(types=['phi', 'psi'])
diheds = xyz.fit_transform_with(featurizer, 'diheds/', fmt='dir-npy')

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
msm = MarkovStateModel(lag_time=20, n_timescales=20, ergodic_cutoff='on')
msm.fit(clustered_trajs)

# Get MFPT
from msmbuilder.tpt import mfpts 
mfpt_matrix = mfpts(msm)

# Get flux matrix
Pi = np.diag(msm.populations_)
Pi = scipy.linalg.fractional_matrix_power(Pi, 1)
Pi_L = scipy.linalg.fractional_matrix_power(Pi, 0.5)
Pi_R = scipy.linalg.fractional_matrix_power(Pi, -0.5)!!
T = msm.transmat_
flux = np.linalg.multi_dot([Pi_L,T,Pi_R])

# Concatenate the trajectories in cluster indices
cluster_indices = np.concatenate(clustered_trajs)

# Combine all trajectories into a trajectory "bag"
temp = xyz[0]
_, num_atoms, num_axis = temp.xyz.shape
reference_frame = temp.slice(0, copy=True)
num_features = num_atoms*num_axis;
pre_X = [np.reshape(traj.xyz, (traj.superpose(reference_frame).xyz.shape[0],num_features)) for traj in xyz]
X = np.concatenate(pre_X)

# Save the msm itself
utils.dump(msm, '/scratch/users/mincheol/' + which_dataset + '/msm_out/msm.pkl')

# Save the raw file 
X.savetxt('/scratch/users/mincheol/' + which_dataset + '/datasets/all_frames.csv')

# save MFPT matrix
mfpt_matrix.dump('/scratch/users/mincheol/' + which_dataset + '/msm_out/msm_mfpt_mat.dat')

# save assignments
cluster_indices.dump('/scratch/users/mincheol/' + which_dataset + '/msm_out/msm_clustering_labels.dat')

# save flux matrix
flux.dump('/scratch/users/mincheol/' + which_dataset + '/msm_out/msm_flux_mat.dat')

# save population vector
msm.populations_.dump('/scratch/users/mincheol/' + which_dataset + '/msm_out/msm_pop_vec.dat')