################################################################################
# Authors: Min Cheol Kim, Christian Choe

# Description: Generates the raw X matrix (rows are frames, cols are XYZ-XYZ-... coordinates) from the stride=10 one

################################################################################

import numpy as np
#strides = [100, 150, 200, 250, 300]
strides = [i*10 for i in range(2, 51)]
for stride in strides:
	full_X = np.load('/scratch/users/mincheol/apo_calmodulin/raw_XYZ_10.dat')
	idx = [i for i in range(full_X.shape[0])]
	sampled_idx = idx[::int(stride/10)]
	sampled_X = full_X[sampled_idx, :]
	sampled_X.dump('/scratch/users/mincheol/apo_calmodulin/raw_XYZ_' + str(stride) + '_.dat')