#!/bin/bash
#SBATCH --mem=16GB
#SBATCH --time=00:10:00
# middleman script for submitting analysis to HPC cluster
# activates right conda environment
# runs script: input 1 is the script, input 2 is the subject, as inherited from
# the runner file.

module load anaconda3
source activate neural_timescales
eval python $1 $2