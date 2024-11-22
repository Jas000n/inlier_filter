#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --account=pr_110_general

module purge


singularity exec --nv \
        --overlay /scratch/sy3913/overlay_images/ParkingWithUncertainty.ext3:rw /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash \
        -c "source /ext3/env.sh; conda activate E2EParking; cd /scratch/sy3913/mynet/dataset; python preprocess.py "
