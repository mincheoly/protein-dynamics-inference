################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Takes in outputs from dimensionality reduction techniques and generates MFPT_plots, plotting distance in reduced
# dimension vs MFPT of frames (estimated by the MFPT between the states they belong in)

# Usage: python clustering_statistics.py -n_neighbors <n_neighbors> -n_components <n_components> -num_clusters <num_clusters> -dataset <dataset label> -technique <method name> -sample_rate <sampling rate>
# n_neighbors is a number of neighbors to use for kernel PCA
# n_components is components in the ISOMAP space
# num_clusters is the number of clusters used for clustering
# dataset is which dataset to use
# method is the dimensionality reduction technique used (e.g. ISOMAP or spectral embedding)

# Example: python clustering_statistics.py -n_neighbors 30 -n_components 40 -num_clusters 97 -dataset fspeptide -technique isomap -sample_rate 0.01

# Output: 1 table, the clustering statistics plot.
# These files are saved with a cluster_comp_<n_neighbors>_<n_components>_<num_clusters>_<dataset>_<method>.png naming scheme.

# Must be editted later to incorporate models.

################################################################################

# Imports
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse as ap
import pandas as pd
from pandas.tools.plotting import table

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

# Load isomap cluster labels
ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(num_clusters) + '_' + str(sample_rate)
iso_cluster = np.load('/scratch/users/mincheol/' + which_dataset + '/' + tech + '_out/' + tech + '_clustering_labels_' + ID + '.dat')

# Load msm cluster labels
# Note: we turned off the ergodic cut off so there are 100 clusters
msm_cluster = np.load('/scratch/users/mincheol/' + which_dataset + '/msm_out/msm_clustering_labels.dat')
msm_pop = np.load('/scratch/users/mincheol/' + which_dataset + '/msm_out/msm_pop_vec.dat')

#find best match for each isomap cluster
#for each isomap cluster it looks for the best msm cluster that fits

#initialize array to store chart data
data = []

# get population estimate for msm and isomap
count = np.zeros([100], dtype=int)
for cluster in msm_cluster:
    count[cluster] += 1
msm_pop_est = count/float(sum(count))

count = np.zeros([100], dtype=int)
for cluster in iso_cluster:
    count[cluster] += 1
iso_pop_est = count/float(sum(count))

# pick the isomap cluster
for m in range(0,100):
    iso_cluster_id = iso_pop_est.argsort()[:][::-1][m]
    iso_frames = np.where(iso_cluster == iso_cluster_id)[0]

    # check overlap of the nth largest cluster for msm and isomap
    for n in range(0,100):
        msm_cluster_id = msm_pop_est.argsort()[:][::-1][n]
        msm_frames = np.where(msm_cluster == msm_cluster_id)[0]

        msm_count = np.zeros([100], dtype=int)
        for idx in iso_frames:
            msm_count[msm_cluster[idx]] += 1
            
        iso_count = np.zeros([100], dtype=int)
        for idx in msm_frames:
            iso_count[iso_cluster[idx]] += 1
        
        max_iso = iso_count[iso_cluster_id]
        max_msm = msm_count[msm_cluster_id]

        if np.argmax(msm_count) == msm_cluster_id:
            percent_msm_in_iso = max_msm/float(sum(msm_count))
            percent_msm_in_iso = round(percent_msm_in_iso,3)
            percent_iso_in_msm = max_iso/float(sum(iso_count))
            percent_iso_in_msm= round(percent_iso_in_msm,3)
            
            #isomap cluster x, msm cluster y, % of isomap cluster in msm cluster, % of msm cluster in isomap cluster, 
            #frame count of isomap cluster x, frame count of msm cluster y, number of frames in common
            temp = [m + 1,n + 1, percent_iso_in_msm, percent_msm_in_iso, sum(msm_count), sum(iso_count), max_iso]
            data.append(temp)


np.set_printoptions(suppress=True)
label = ['iso #', 'msm #', '%iso->msm', '%msm->iso', 'total iso', 'total msm', 'common']
#print label
#print np.array(data)

df = pd.DataFrame(data, columns=label)
df.to_csv('/scratch/users/mincheol/' + which_dataset + '/tables/' + tech + '_cluster_comp_' + ID + '.csv');
print('done saving')
