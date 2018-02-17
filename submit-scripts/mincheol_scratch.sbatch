#!/bin/bash

#SBATCH --job-name=mincheol_scratch
#SBATCH --output=/scratch/users/mincheol/job_outputs/mincheol_scratch.%j.out
#SBATCH --error=/scratch/users/mincheol/job_outputs/mincheol_scratch.%j.err
#SBATCH --time=2:00:00
#SBATCH --qos=dev
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=mincheol@stanford.edu

# otherwise: 
module load anaconda
source activate test_env
python population_sample.py -num_clusters 100 -dataset apo_calmodulin

# python generate_subsamples.py
source deactivate
