#!/bin/bash

#SBATCH --job-name=change_angle
#SBATCH --output=/scratch/users/cachoe/job_outputs/change_angle.%j.out
#SBATCH --error=/scratch/users/cachoe/job_outputs/change_angle.%j.err
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
python change_angle.py -dataset fspeptide
python change_angle.py -dataset apo_calmodulin
source deactivate