################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Computes the PCA in reduced dimension space

# Usage: python fit_model.py -method <method> -n_neighbors <n_neighbors> -n_components <n_components> -dataset <dataset label> -sample_rate <sampling rate> -pc <pc>
# method is the method to use. Currently: KernelPCA, Isomap, and SpectralEmbedding
# n_neighbors is a number of neighbors to use
# n_components is components in the ISOMAP space
# dataset is which dataset to use
# sample_rate is the subsampling rate
# pc is the number of principal components you want. 

# Example: python perform_PCA.py -method isomap -n_neighbors 30 -n_components 40 -dataset fspeptide -stride 100 -pc 4

# Output: A numpy dat file containing the PC components. 
# These files are saved with a X_<method>_<n_neighbors>_<n_components>_<sample_rate>_pc.dat naming scheme.

################################################################################

# imports
import numpy as np
import argparse as ap
from sklearn.decomposition import PCA

# Set up ArgumentParser
parser = ap.ArgumentParser(description='Apply dimension reduction script.')
parser.add_argument('-method', action="store", dest='method')
parser.add_argument('-n_components', action="store", dest='n_components', type=int)
parser.add_argument('-n_neighbors', action="store", dest='n_neighbors', type=int)
parser.add_argument('-dataset', action='store', dest='which_dataset')
parser.add_argument('-stride', action='store', dest='stride', type=int)
parser.add_argument('-pc', action="store", dest='pc', type=int)
args = parser.parse_args()

# Assign argument variables
method = args.method
n_neighbors = args.n_neighbors
n_components = args.n_components
which_dataset = args.which_dataset
stride = args.stride
pc = args.pc

# Combine n_neighbors and n_components to produce an ID
ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(stride)

# Load things
X_rd = np.load('/scratch/users/mincheol/' + which_dataset + '/reduced_dimension/X_'+ method +'_' + ID + '.dat')

# Remove NaNs
indices = []
for i in range(X_rd.shape[0]):
    if np.isnan(np.sum(X_rd[i,:])) or np.isinf(np.sum(X_rd[i,:])) or np.sum(X_rd[i,:]) > 100000 or np.sum(X_rd[i,:]) < 100000:
        indices.append(i)

X_rd = np.delete(X_rd, indices, 0)

print(np.sum(np.sum(X_rd)))

# Perform PCA
pca = PCA(n_components=pc)
X_rp = pca.fit_transform(X_rd)

# Save the PC coordinates
X_rp.dump('/scratch/users/mincheol/' + which_dataset + '/principal_components/X_pc_'+ method +'_' + ID + '_' + str(pc) + '.dat')