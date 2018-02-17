################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Min Cheol's scratch pad python file

################################################################################

# get data
import numpy as np
from sklearn.decomposition import PCA

# for i in range(25, 46):
# 	stride = i*10
# 	filename = '/scratch/users/mincheol/apo_calmodulin/datasets/indices_' + str(stride) + '.dat'
# 	indices = np.load(filename)
# 	print(stride)
# 	print(indices.shape)

n_neighbors_set = [40]
n_components_set = [40]
#strides = [5000, 15000, 25000] #apo_calmodulin
#strides = [3000, 9000, 15000] #fspeptide
datasets = ['fspeptide']
#datasets = ['apo_calmodulin']
methods = ['isomap']
pc = 4
sampling_range = np.arange(1000, 19001, 500)
sampling_range = np.concatenate((np.arange(100, 1000, 100), sampling_range))
sampling_range = np.append(sampling_range, 28000)
sampling_range = sampling_range[::-1]
strides = sampling_range

data_type = ['angle'][0]
raw = False

for n_components in n_components_set:
	for which_dataset in datasets:
		for n_neighbors in n_neighbors_set:
			for stride in strides:
				for method in methods:
					print (n_neighbors,stride,method)
					ID = str(n_neighbors) + '_' + str(n_components) + '_' + str(stride)

					# Load things
					if raw:
						X_rd = np.loadtxt('/scratch/users/mincheol/' + which_dataset + '/sim_datasets/raw_'+ data_type +'_' + str(stride) + '.csv', delimiter=',')
						print('open')
					else:
						X_rd = np.loadtxt('/scratch/users/mincheol/' + which_dataset + '/reduced_dimension/X_'+ method +'_' + ID + '.csv', delimiter=',')

					# Remove NaNs
					indices = []
					for i in range(X_rd.shape[0]):
					    if np.isnan(np.sum(X_rd[i,:])) or np.isinf(np.sum(X_rd[i,:])) or np.sum(X_rd[i,:]) > 150 or np.sum(X_rd[i,:]) < -150:
					        indices.append(i)

					X_rd = np.delete(X_rd, indices, 0)

					# Perform PCA
					pca = PCA(n_components=pc)
					X_rp = pca.fit_transform(X_rd)

					# Save the PC coordinates
					if raw:
						X_rp.dump('/scratch/users/mincheol/' + which_dataset + '/principal_components/X_pc_raw_' + str(stride) + '_' + str(pc) + '.dat')
					else:
						X_rp.dump('/scratch/users/mincheol/' + which_dataset + '/principal_components/X_pc_'+ method +'_' + ID + '_' + str(pc) + '.dat')