#!/bin/bash

#SBATCH --job-name=rd_pipeline_kpca
#SBATCH --output=/scratch/users/mincheol/job_outputs/rd_pipeline_kpca.%j.out
#SBATCH --error=/scratch/users/mincheol/job_outputs/rd_pipeline_kpca.%j.err
#SBATCH --time=3:00:00
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --mem=60GB
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mincheol@stanford.edu

# otherwise: 
module load anaconda
source activate test_env
python fit_model.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 300
python apply_dr.py -method kernelPCA -n_neighbors 40 -n_components 10 -dataset apo_calmodulin -stride 300
source deactivate
