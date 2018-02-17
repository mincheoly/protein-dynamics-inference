################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Applies a dimensionality reduction on a given dataset. 

# Usage: python isomap_clustering.py -method <method> -n_neighbors <n_neighbors> -n_components <n_components> -dataset <dataset label>
# method is the method to use. Currently: KernelPCA, Isomap, and SpectralEmbedding
# n_neighbors is a number of neighbors to use
# n_components is components in the ISOMAP space
# dataset is which dataset to use

# Output: 1 .csv file containing the transformed coordinates.
# These files are saved with a *_<n_neighbors>_<n_components>_<frames>.dat naming scheme.

################################################################################

# imports
import numpy as np
import argparse as ap
from sklearn import manifold
from sklearn import decomposition
from sklearn.externals import joblib

# additional imports
import tempfile
import os
os.chdir(tempfile.mkdtemp())

# Set up ArgumentParser
parser = ap.ArgumentParser(description='Apply dimension reduction script.')
parser.add_argument('-method', action="store", dest='method')
parser.add_argument('-n_components', action="store", dest='n_components', type=int)
parser.add_argument('-n_neighbors', action="store", dest='n_neighbors', type=int)
parser.add_argument('-dataset', action='store', dest='which_dataset')
parser.add_argument('-frames', action='store', dest='frames', type=int)
parser.add_argument('-feature', action='store', dest='feature')
args = parser.parse_args()

# Assign argument variables
method = args.method
n_neighbors = args.n_neighbors
n_components = args.n_components
which_dataset = args.which_dataset
frames = args.frames
feature = args.feature

# List of frame numbers to run
if which_dataset == 'fspeptide':
	sampling_range = np.arange(1000, 19001, 500)
	sampling_range = np.concatenate((np.arange(100, 1000, 100), sampling_range))
	sampling_range = np.append(sampling_range, 28000)
if which_dataset == 'apo_calmodulin':
	sampling_range = np.arange(20000, 30001, 2000)
	sampling_range = np.concatenate((np.arange(5000, 20000, 1000), sampling_range))
	sampling_range = np.concatenate((np.arange(1000, 5000, 500), sampling_range))
	sampling_range = np.concatenate((np.arange(200, 1000, 200), sampling_range))
	#sampling_range = np.append(sampling_range, 353566)
#sampling_range = sampling_range[::-1]

for frames in sampling_range:
	# Combine n_neighbors and n_components to produce an ID
	ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(frames)

	# Load appropriate X matrix
	X = np.loadtxt('/scratch/users/mincheol/' + which_dataset + '/sim_datasets/raw_'+feature+'_' + str(frames) + '.csv', delimiter=',')
	print('Raw data loaded')

	# Load the appropriate model 
	if which_dataset == 'holo_calmodulin':
		model = joblib.load('/scratch/users/mincheol/apo_calmodulin/models/' + method + '_model_' + ID + '.pkl')
	else:
		model = joblib.load('/scratch/users/mincheol/' + which_dataset + '/models/' + method + '_model_' + ID + '.pkl')
	print('Model loaded')

	# Transform X matrix in batches
	X_rd = np.empty((X.shape[0], n_components))
	num_batches = 1000
	batch_size = int(X.shape[0]/num_batches) # size of each batch
	for batch in range(num_batches+1):
		start_idx = batch * batch_size
		end_idx = (batch + 1)*batch_size if (batch + 1)*batch_size < X.shape[0] else X.shape[0]
		if start_idx != end_idx:
			X_rd[start_idx:end_idx, :] = model.transform(X[start_idx:end_idx,:])

	# Saved X in reduced dimension
	np.savetxt('/scratch/users/mincheol/' + which_dataset + '/reduced_dimension/X_'+ method +'_' + ID + '.csv', X_rd, delimiter=',')
	print('Coordinates saved in reduced dimension')

