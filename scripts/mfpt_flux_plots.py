################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Takes in outputs from dimensionality reduction techniques and generates MFPT_plots, plotting distance in reduced
# dimension vs MFPT of frames (estimated by the MFPT between the states they belong in)

# Usage: python mfpt_plots.py -n_neighbors <n_neighbors> -n_components <n_components> -num_clusters <num_clusters> -dataset <dataset label> -method <method name> -sample_rate <sampling rate>
# n_neighbors is a number of neighbors to use for kernel PCA
# n_components is components in the ISOMAP space
# num_clusters is the number of clusters used for clustering
# dataset is which dataset to use
# method is the dimensionality reduction technique used (e.g. ISOMAP or spectral embedding)

# Example: python mfpt_flux_plots.py -n_neighbors 30 -n_components 40 -num_clusters 97 -dataset fspeptide -technique isomap -sample_rate 0.1

# Output: 1 file, the MFPT plot.
# These files are saved with a mfpt_<n_neighbors>_<n_components>_<num_clusters>_<dataset>_<method>.png naming scheme.


################################################################################

# Imports
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse as ap

# Set up ArgumentParser
parser = ap.ArgumentParser(description='ISOMAP processing script.')
parser.add_argument('-n_components', action="store", dest='n_components', type=int)
parser.add_argument('-n_neighbors', action="store", dest='n_neighbors', type=int)
parser.add_argument('-num_clusters', action="store", dest='num_clusters', type=int, default=97)
parser.add_argument('-dataset', action='store', dest='which_dataset')
parser.add_argument('-technique', action='store', dest='tech')
parser.add_argument('-sample_rate', action='store', dest='sample_rate', type=float)
args = parser.parse_args()

# Assign argument variables
n_neighbors = args.n_neighbors
n_components = args.n_components
num_clusters = args.num_clusters
which_dataset = args.which_dataset
tech = args.tech
sample_rate = args.sample_rate

# Load the dimensionality reduction results
ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(num_clusters) + '_' + str(sample_rate)
rd_coors = np.load('/scratch/users/mincheol/' + which_dataset + '/' + tech + '_out/' + tech + '_coordinates_' + ID + '.dat')
rd_cluster_labels = np.load('/scratch/users/mincheol/' + which_dataset + '/' + tech + '_out/' + tech + '_coordinates_' + ID + '.dat')

# Import the MSM builder results
msm_cluster_labels = np.load('/scratch/users/mincheol/' + which_dataset + '/msm_out/msm_clustering_labels.dat')
msm_mfpt_mat = np.load('/scratch/users/mincheol/' + which_dataset + '/msm_out/msm_mfpt_mat.dat')
msm_flux_mat = np.load('/scratch/users/mincheol/' + which_dataset + '/msm_out/msm_flux_mat.dat')

msm_mfpt_sym = (msm_mfpt_mat + msm_mfpt_mat.transpose())/2
num_frames = rd_cluster_labels.shape[0]

# select frames to compare to each other
indices = [i for i in range(num_frames)]
np.random.shuffle(indices)
indices = indices[:100]

rd_dist = []
mfpt = []
flux = []

# Collect ISOMAP distances and MFTPS
for (idx1, frame1) in enumerate(indices):
    for (idx2, frame2) in enumerate(indices[(idx1+1):]):
        if msm_cluster_labels[frame1] is not msm_cluster_labels[frame2]:
            rd_dist.append( np.linalg.norm(rd_coors[frame1, :] - rd_coors[frame2, :])  )
            mfpt.append(msm_mfpt_sym[msm_cluster_labels[frame2]][msm_cluster_labels[frame1]])
            flux.append(msm_flux_mat[msm_cluster_labels[frame2]][msm_cluster_labels[frame1]])

with np.errstate(divide='ignore'):
	plt.scatter(np.array(rd_dist), np.log(np.array(mfpt)))
plt.xlabel('Reduced dimension distance')
plt.ylabel('MFPT')
plt.savefig('/scratch/users/mincheol/' + which_dataset + '/plots/' + tech + '_mfpt_' + ID + '.png');
plt.clf()

plt.scatter(np.array(rd_dist), np.array(flux))
plt.xlabel('Reduced dimension distance')
plt.ylabel('Flux')
plt.savefig('/scratch/users/mincheol/' + which_dataset + '/plots/' + tech + '_flux' + ID + '.png');
plt.clf()

print('done saving')