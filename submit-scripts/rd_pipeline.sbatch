#!/bin/bash

#SBATCH --job-name=rd_pipeline
#SBATCH --output=/scratch/users/cachoe/job_outputs/rd_pipeline.%j.out
#SBATCH --error=/scratch/users/cachoe/job_outputs/rd_pipeline.%j.err
#SBATCH --time=16:00:00
#SBATCH --qos=bigmem
#SBATCH -p bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=100GB
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=cachoe@stanford.edu

# otherwise:
module load anaconda
source activate test_env
python fit_model.py -method isomap -n_neighbors 40 -n_components 40 -dataset fspeptide -frames 0 -feature phi_psi
python apply_dr.py -method isomap -n_neighbors 40 -n_components 40 -dataset fspeptide -frames 0 -feature phi_psi

python fit_model.py -method isomap -n_neighbors 40 -n_components 40 -dataset apo_calmodulin -frames 0 -feature phi_psi
python apply_dr.py -method isomap -n_neighbors 40 -n_components 40 -dataset apo_calmodulin -frames 0 -feature phi_psi
source deactivate